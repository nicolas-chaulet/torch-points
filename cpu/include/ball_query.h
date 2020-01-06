#pragma once
#include <torch/extension.h>
std::pair<at::Tensor, at::Tensor> ball_query(at::Tensor query,
					     at::Tensor support,
					     float radius, int max_num=-1, int mode=0);

std::pair<at::Tensor, at::Tensor> batch_ball_query(at::Tensor query,
						   at::Tensor support,
						   at::Tensor query_batch,
						   at::Tensor support_batch,
						   float radius, int max_num=-1, int mode=0);
