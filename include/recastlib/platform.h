#ifndef RECASTLIB_PLATFORM_H
#define RECASTLIB_PLATFORM_H

// CMake publishes this macro through the recastlib target. Keep the fallback
// disabled because adapter availability depends on both the host platform and
// the RECASTLIB_WITH_LINUX_AIO build option, not merely on __linux__.
#ifndef RECASTLIB_HAS_LINUX_AIO
#define RECASTLIB_HAS_LINUX_AIO 0
#endif

#endif  // RECASTLIB_PLATFORM_H
