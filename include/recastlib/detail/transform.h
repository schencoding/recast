#ifndef RECASTLIB_DETAIL_TRANSFORM_H
#define RECASTLIB_DETAIL_TRANSFORM_H

#include <cstddef>
#include <memory>

#include <faiss/VectorTransform.h>

namespace recastlib::detail {

/**
 * Trains an owning full-dimensional PCA rotation from row-major vectors.
 *
 * @param dimension Input and output dimension D.
 * @param count Number of training vectors.
 * @param vectors Contiguous buffer containing count * dimension floats.
 * @return Trained D-by-D linear transform.
 */
std::unique_ptr<faiss::LinearTransform> train_pca_rotation(
    size_t dimension, size_t count, const float* vectors);

}  // namespace recastlib::detail

#endif  // RECASTLIB_DETAIL_TRANSFORM_H
