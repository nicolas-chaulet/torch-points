#include "ball_query.h"
#include "chamfer_dist.h"
#include "cubic_feature_sampling.h"
#include "gridding.h"
#include "gridding_reverse.h"
#include "interpolate.h"
#include "metrics.h"
#include "sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("furthest_point_sampling", &furthest_point_sampling);

    m.def("three_nn", &three_nn);
    m.def("three_interpolate", &three_interpolate);
    m.def("three_interpolate_grad", &three_interpolate_grad);

    m.def("ball_query_dense", &ball_query_dense);
    m.def("ball_query_partial_dense", &ball_query_partial_dense);

    m.def("instance_iou_cuda", &instance_iou_cuda);

    m.def("chamfer_dist", &chamfer_dist);
    m.def("chamfer_dist_grad", &chamfer_dist_grad);

    m.def("cubic_feature_sampling", &cubic_feature_sampling);
    m.def("cubic_feature_sampling_grad", &cubic_feature_sampling_grad);

    m.def("gridding", &gridding);
    m.def("gridding_grad", &gridding_grad);

    m.def("gridding_reverse", &gridding_reverse);
    m.def("gridding_reverse_grad", &gridding_reverse_grad);
}
