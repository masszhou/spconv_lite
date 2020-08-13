// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SPCONV_UTILS_NMS_CUH
#define SPCONV_UTILS_NMS_CUH

template <typename DType, int BLOCK_THREADS>
int _nms_gpu(int *keep_out, const DType *boxes_host, int boxes_num,
             int boxes_dim, DType nms_overlap_thresh, int device_id);

#endif //SPCONV_UTILS_NMS_CUH