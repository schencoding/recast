#include "list_file_aio_refiner.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unistd.h>
#include <utility>

#if RECASTLIB_HAS_LINUX_AIO
#include <libaio.h>
#endif

#include "external_io_util.h"

namespace recastlib::adapters {
namespace {

using Clock = std::chrono::steady_clock;

struct ExactPairWorseFirst {
  bool operator()(const Neighbor& lhs, const Neighbor& rhs) const {
    if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
    return lhs.id < rhs.id;
  }
};

using ExactHeap =
    std::priority_queue<Neighbor, std::vector<Neighbor>, ExactPairWorseFirst>;

void insert_exact(ExactHeap* heap, size_t topk, uint32_t id, float distance) {
  if (!std::isfinite(distance) || distance < 0.0f) {
    throw std::runtime_error("exact distance function returned an invalid value");
  }
  heap->push(Neighbor{id, distance});
  if (heap->size() > topk) heap->pop();
}

std::vector<Neighbor> finish_topk(ExactHeap* heap) {
  std::vector<Neighbor> output;
  output.reserve(heap->size());
  while (!heap->empty()) {
    output.push_back(heap->top());
    heap->pop();
  }
  std::reverse(output.begin(), output.end());
  return output;
}

uint64_t elapsed_nanoseconds(Clock::time_point begin) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin)
          .count());
}

struct CandidateInRequest {
  size_t candidate_index = 0;
  size_t record_in_buffer = 0;
};

struct ReadPlan {
  uint32_t list_no = 0;
  uint64_t aligned_offset = 0;
  size_t request_bytes = 0;
  size_t page_count = 0;
  std::vector<CandidateInRequest> candidates;
};

struct OpenFile {
  uint32_t list_no = 0;
  int fd = -1;
};

struct Location {
  uint32_t list_no = 0;
  uint64_t page_no = 0;
  size_t candidate_index = 0;
  size_t record_in_page = 0;
};

std::vector<ReadPlan> build_read_plans(
    const ListFileRecordStore& store,
    const std::vector<ListCandidate>& candidates) {
  std::vector<Location> locations;
  locations.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    const uint64_t page_no = candidates[i].offset / store.records_per_page();
    const size_t slot = candidates[i].offset % store.records_per_page();
    locations.push_back(Location{
        candidates[i].list_no,
        page_no,
        i,
        slot * store.record_stride()});
  }
  std::sort(locations.begin(), locations.end(), [](const auto& lhs,
                                                   const auto& rhs) {
    if (lhs.list_no != rhs.list_no) return lhs.list_no < rhs.list_no;
    if (lhs.page_no != rhs.page_no) return lhs.page_no < rhs.page_no;
    return lhs.candidate_index < rhs.candidate_index;
  });

  std::vector<ReadPlan> plans;
  uint32_t previous_list = UINT32_MAX;
  uint64_t previous_page = std::numeric_limits<uint64_t>::max();
  for (const Location& location : locations) {
    const bool same_list = location.list_no == previous_list;
    const bool same_page = same_list && location.page_no == previous_page;
    if (!same_page) {
      const bool adjacent = same_list && previous_page !=
              std::numeric_limits<uint64_t>::max() &&
          location.page_no == previous_page + 1;
      if (!adjacent) {
        const size_t byte_offset = external_io_detail::checked_multiply(
            static_cast<size_t>(location.page_no), store.page_size(),
            "list-file read offset overflows");
        if (byte_offset >
            static_cast<size_t>(std::numeric_limits<off_t>::max())) {
          throw std::overflow_error("list-file read offset exceeds off_t");
        }
        plans.push_back(ReadPlan{
            location.list_no, static_cast<uint64_t>(byte_offset), 0, 0, {}});
      }
      ReadPlan& plan = plans.back();
      ++plan.page_count;
      plan.request_bytes = external_io_detail::checked_multiply(
          plan.page_count, store.page_size(),
          "merged list-file request size overflows");
      previous_list = location.list_no;
      previous_page = location.page_no;
    }

    ReadPlan& plan = plans.back();
    const uint64_t first_page = plan.aligned_offset / store.page_size();
    const size_t page_in_request =
        static_cast<size_t>(location.page_no - first_page);
    plan.candidates.push_back(CandidateInRequest{
        location.candidate_index,
        page_in_request * store.page_size() + location.record_in_page});
  }
  return plans;
}

std::vector<uint32_t> distinct_lists(const std::vector<ReadPlan>& plans) {
  std::vector<uint32_t> lists;
  for (const ReadPlan& plan : plans) {
    if (lists.empty() || lists.back() != plan.list_no) {
      lists.push_back(plan.list_no);
    }
  }
  return lists;
}

void close_files(std::vector<OpenFile>* files, ListFileRefineStats* stats) {
  const auto begin = Clock::now();
  int first_error = 0;
  for (OpenFile& file : *files) {
    if (file.fd >= 0) {
      if (::close(file.fd) != 0 && first_error == 0) first_error = errno;
      file.fd = -1;
    }
  }
  if (stats != nullptr) stats->close_nanoseconds += elapsed_nanoseconds(begin);
  if (first_error != 0) {
    throw external_io_detail::system_error("close(list file)", first_error);
  }
}

std::vector<OpenFile> open_files(
    const ListFileRecordStore& store,
    const std::vector<uint32_t>& lists,
    size_t begin,
    size_t end,
    ListFileRefineStats* stats) {
  std::vector<OpenFile> files;
  files.reserve(end - begin);
  const auto open_begin = Clock::now();
  try {
    for (size_t i = begin; i < end; ++i) {
      const std::string path = store.list_path(lists[i]);
      const int fd = ::open(
          path.c_str(), O_RDONLY | O_CLOEXEC |
              external_io_detail::direct_flag(store.direct_io_enabled()));
      if (fd < 0) throw external_io_detail::system_error("open(" + path + ")");
      files.push_back(OpenFile{lists[i], fd});
    }
  } catch (...) {
    for (OpenFile& file : files) {
      if (file.fd >= 0) ::close(file.fd);
    }
    throw;
  }
  if (stats != nullptr) {
    stats->opened_files += files.size();
    stats->open_nanoseconds += elapsed_nanoseconds(open_begin);
  }
  return files;
}

void process_plan(
    const uint8_t* bytes,
    const ReadPlan& plan,
    const float* query,
    size_t dimension,
    size_t topk,
    size_t record_stride,
    size_t payload_bytes,
    ExactDistanceFunction distance_function,
    ExactHeap* exact_top,
    ListFileRefineStats* stats) {
  for (const CandidateInRequest& item : plan.candidates) {
    if (item.record_in_buffer > plan.request_bytes ||
        record_stride > plan.request_bytes - item.record_in_buffer ||
        sizeof(uint32_t) + payload_bytes > record_stride) {
      throw std::logic_error("candidate record is outside its read request");
    }
    uint32_t logical_id = 0;
    std::memcpy(&logical_id, bytes + item.record_in_buffer, sizeof(logical_id));
    const void* payload = bytes + item.record_in_buffer + sizeof(logical_id);
    insert_exact(
        exact_top, topk, logical_id,
        distance_function(query, payload, dimension));
  }
  if (stats != nullptr) stats->refined_vectors += plan.candidates.size();
}

void account_request(const ReadPlan& plan, ListFileRefineStats* stats) {
  if (stats == nullptr) return;
  ++stats->read_requests;
  stats->unique_pages += plan.page_count;
  stats->requested_bytes += plan.request_bytes;
}

int descriptor_for(
    const std::vector<OpenFile>& files, uint32_t list_no) {
  const auto found = std::lower_bound(
      files.begin(), files.end(), list_no,
      [](const OpenFile& lhs, uint32_t rhs) { return lhs.list_no < rhs; });
  if (found == files.end() || found->list_no != list_no) {
    throw std::logic_error("read plan references an unopened list file");
  }
  return found->fd;
}

void refine_blocking(
    const ListFileRecordStore& store,
    const std::vector<ReadPlan>& plans,
    size_t plan_begin,
    size_t plan_end,
    const std::vector<OpenFile>& files,
    const float* query,
    size_t dimension,
    size_t topk,
    ExactDistanceFunction distance_function,
    ExactHeap* exact_top,
    ListFileRefineStats* stats) {
  size_t capacity = store.page_size();
  external_io_detail::AlignedBuffer buffer(store.page_size(), capacity);
  for (size_t i = plan_begin; i < plan_end; ++i) {
    const ReadPlan& plan = plans[i];
    if (capacity < plan.request_bytes) {
      capacity = plan.request_bytes;
      buffer = external_io_detail::AlignedBuffer(store.page_size(), capacity);
    }
    const int fd = descriptor_for(files, plan.list_no);
    const auto wait_begin = Clock::now();
    ssize_t result;
    do {
      result = ::pread(
          fd, buffer.data(), plan.request_bytes,
          static_cast<off_t>(plan.aligned_offset));
    } while (result < 0 && errno == EINTR);
    if (stats != nullptr) {
      stats->wait_nanoseconds += elapsed_nanoseconds(wait_begin);
      stats->peak_inflight = std::max<size_t>(stats->peak_inflight, 1);
    }
    if (result < 0) throw external_io_detail::system_error("pread(list file)");
    if (static_cast<size_t>(result) != plan.request_bytes) {
      throw std::runtime_error("short read from list file");
    }
    account_request(plan, stats);
    process_plan(
        static_cast<const uint8_t*>(buffer.data()), plan, query, dimension,
        topk, store.record_stride(), store.payload_bytes(),
        distance_function, exact_top, stats);
  }
}

#if RECASTLIB_HAS_LINUX_AIO
struct AioSlot {
  explicit AioSlot(size_t alignment)
      : buffer(alignment, alignment), capacity(alignment) {}

  external_io_detail::AlignedBuffer buffer;
  size_t capacity = 0;
  iocb control{};
  size_t plan_index = 0;
};

/**
 * One Linux AIO context per query-worker thread.
 *
 * A context is independent of the list files used by a particular query, so
 * recreating it in every search only adds kernel setup/teardown overhead.  The
 * context is therefore retained by the OpenMP worker and reused by subsequent
 * queries on that same thread.  File descriptors and request buffers remain
 * query-local; only the empty completion-queue object survives between calls.
 */
class ThreadLocalAioContext {
 public:
  ~ThreadLocalAioContext() { reset(); }

  io_context_t acquire(size_t queue_depth) {
    if (context_ != 0 && capacity_ >= queue_depth) return context_;
    reset();
    const int result = ::io_setup(static_cast<int>(queue_depth), &context_);
    if (result < 0) {
      context_ = 0;
      throw external_io_detail::system_error("io_setup", -result);
    }
    capacity_ = queue_depth;
    return context_;
  }

  void reset() noexcept {
    if (context_ != 0) ::io_destroy(context_);
    context_ = 0;
    capacity_ = 0;
  }

 private:
  io_context_t context_ = 0;
  size_t capacity_ = 0;
};

thread_local ThreadLocalAioContext worker_aio_context;

void ensure_capacity(AioSlot* slot, size_t alignment, size_t bytes) {
  if (slot->capacity >= bytes) return;
  slot->buffer = external_io_detail::AlignedBuffer(alignment, bytes);
  slot->capacity = bytes;
}

void prepare_aio(
    AioSlot* slot,
    const ReadPlan& plan,
    size_t plan_index,
    int fd,
    size_t alignment) {
  ensure_capacity(slot, alignment, plan.request_bytes);
  slot->plan_index = plan_index;
  std::memset(&slot->control, 0, sizeof(slot->control));
  ::io_prep_pread(
      &slot->control, fd, slot->buffer.data(), plan.request_bytes,
      static_cast<off_t>(plan.aligned_offset));
  slot->control.data = slot;
}

void submit_requests(
    io_context_t context,
    std::vector<iocb*>* requests,
    const std::vector<ReadPlan>& plans,
    ListFileRefineStats* stats) {
  size_t submitted = 0;
  while (submitted < requests->size()) {
    const int result = ::io_submit(
        context, static_cast<long>(requests->size() - submitted),
        requests->data() + submitted);
    if (result < 0) {
      throw external_io_detail::system_error("io_submit", -result);
    }
    if (result == 0) throw std::runtime_error("io_submit made no progress");
    for (int i = 0; i < result; ++i) {
      const auto* slot = static_cast<const AioSlot*>(
          (*requests)[submitted + static_cast<size_t>(i)]->data);
      account_request(plans[slot->plan_index], stats);
    }
    submitted += static_cast<size_t>(result);
  }
}

void refine_aio(
    io_context_t context,
    const ListFileRecordStore& store,
    const std::vector<ReadPlan>& plans,
    size_t plan_begin,
    size_t plan_end,
    const std::vector<OpenFile>& files,
    size_t queue_depth,
    const float* query,
    size_t dimension,
    size_t topk,
    ExactDistanceFunction distance_function,
    ExactHeap* exact_top,
    ListFileRefineStats* stats) {
  std::vector<AioSlot> slots;
  slots.reserve(queue_depth);
  for (size_t i = 0; i < queue_depth; ++i) {
    slots.emplace_back(store.page_size());
  }
  const size_t plan_count = plan_end - plan_begin;
  const size_t initial = std::min(queue_depth, plan_count);
  size_t next_plan = plan_begin;
  std::vector<iocb*> submissions;
  submissions.reserve(queue_depth);
  for (size_t i = 0; i < initial; ++i) {
    const ReadPlan& plan = plans[next_plan];
    prepare_aio(
        &slots[i], plan, next_plan, descriptor_for(files, plan.list_no),
        store.page_size());
    submissions.push_back(&slots[i].control);
    ++next_plan;
  }
  submit_requests(context, &submissions, plans, stats);
  size_t inflight = initial;
  if (stats != nullptr) {
    stats->peak_inflight = std::max(stats->peak_inflight, inflight);
  }

  std::vector<io_event> events(queue_depth);
  while (inflight != 0) {
    const auto wait_begin = Clock::now();
    int completed;
    do {
      completed = ::io_getevents(
          context, 1, static_cast<long>(events.size()), events.data(), nullptr);
    } while (completed == -EINTR);
    if (stats != nullptr) {
      stats->wait_nanoseconds += elapsed_nanoseconds(wait_begin);
    }
    if (completed < 0) {
      throw external_io_detail::system_error("io_getevents", -completed);
    }
    if (completed == 0) {
      throw std::runtime_error("io_getevents made no progress");
    }
    submissions.clear();
    for (int i = 0; i < completed; ++i) {
      auto* slot = static_cast<AioSlot*>(events[static_cast<size_t>(i)].data);
      if (slot == nullptr) {
        throw std::runtime_error("AIO completion lost its request slot");
      }
      const ReadPlan& plan = plans[slot->plan_index];
      const int64_t result = events[static_cast<size_t>(i)].res;
      if (events[static_cast<size_t>(i)].res2 != 0) {
        throw std::runtime_error("O_DIRECT list read returned nonzero res2");
      }
      if (result < 0) {
        throw external_io_detail::system_error(
            "O_DIRECT list read", static_cast<int>(-result));
      }
      if (static_cast<size_t>(result) != plan.request_bytes) {
        throw std::runtime_error("short O_DIRECT read from list file");
      }
      process_plan(
          static_cast<const uint8_t*>(slot->buffer.data()), plan, query,
          dimension, topk, store.record_stride(), store.payload_bytes(),
          distance_function, exact_top, stats);
      --inflight;

      if (next_plan < plan_end) {
        const ReadPlan& next = plans[next_plan];
        prepare_aio(
            slot, next, next_plan, descriptor_for(files, next.list_no),
            store.page_size());
        submissions.push_back(&slot->control);
        ++next_plan;
      }
    }
    if (!submissions.empty()) {
      submit_requests(context, &submissions, plans, stats);
      inflight += submissions.size();
      if (stats != nullptr) {
        stats->peak_inflight = std::max(stats->peak_inflight, inflight);
      }
    }
  }
}
#endif

}  // namespace

ListFileAioRefiner::ListFileAioRefiner(
    const ListFileRecordStore& store,
    ListFileRefineOptions options)
    : store_(store), options_(options) {
  if (options_.queue_depth == 0 ||
      options_.max_open_files < options_.queue_depth) {
    throw std::invalid_argument("invalid list-file refinement options");
  }
#if !RECASTLIB_HAS_LINUX_AIO
  if (options_.queue_depth != 1) {
    throw std::invalid_argument(
        "queue_depth greater than one requires a Linux libaio build");
  }
#endif
}

std::vector<Neighbor> ListFileAioRefiner::refine(
    const float* query,
    size_t dimension,
    const std::vector<ListCandidate>& candidates,
    size_t topk,
    ExactDistanceFunction distance_function,
    ListFileRefineStats* stats) const {
  if (query == nullptr || dimension == 0 || topk == 0 ||
      candidates.size() < topk || distance_function == nullptr) {
    throw std::invalid_argument("invalid list-file refinement request");
  }
  for (const ListCandidate& candidate : candidates) {
    if (candidate.list_no >= store_.nlist()) {
      throw std::out_of_range("refinement candidate list is invalid");
    }
  }

  const std::vector<ReadPlan> plans = build_read_plans(store_, candidates);
  const std::vector<uint32_t> lists = distinct_lists(plans);
  ExactHeap exact_top;
#if RECASTLIB_HAS_LINUX_AIO
  io_context_t aio_context = 0;
  if (options_.queue_depth > 1) {
    aio_context = worker_aio_context.acquire(options_.queue_depth);
  }
#endif
  size_t list_begin = 0;
  size_t plan_begin = 0;
  try {
    while (list_begin < lists.size()) {
      const size_t list_end = std::min(
          lists.size(), list_begin + options_.max_open_files);
      std::vector<OpenFile> files =
          open_files(store_, lists, list_begin, list_end, stats);
      size_t plan_end = plan_begin;
      while (plan_end < plans.size() &&
             plans[plan_end].list_no <= lists[list_end - 1]) {
        ++plan_end;
      }
      try {
        if (options_.queue_depth == 1) {
          refine_blocking(
              store_, plans, plan_begin, plan_end, files, query, dimension,
              topk, distance_function, &exact_top, stats);
        } else {
#if RECASTLIB_HAS_LINUX_AIO
          refine_aio(
              aio_context, store_, plans, plan_begin, plan_end, files,
              options_.queue_depth, query, dimension, topk, distance_function,
              &exact_top, stats);
#else
          throw std::logic_error("unreachable non-Linux AIO refinement path");
#endif
        }
        close_files(&files, stats);
      } catch (...) {
#if RECASTLIB_HAS_LINUX_AIO
        // io_destroy first cancels/waits for any submitted operations. Closing a
        // descriptor while its iocbs still reference it would violate the AIO
        // lifetime contract on the exceptional path.
        if (aio_context != 0) {
          worker_aio_context.reset();
          aio_context = 0;
        }
#endif
        for (OpenFile& file : files) {
          if (file.fd >= 0) ::close(file.fd);
        }
        throw;
      }
      plan_begin = plan_end;
      list_begin = list_end;
    }
  } catch (...) {
#if RECASTLIB_HAS_LINUX_AIO
    if (aio_context != 0) worker_aio_context.reset();
#endif
    throw;
  }
  return finish_topk(&exact_top);
}

}  // namespace recastlib::adapters
