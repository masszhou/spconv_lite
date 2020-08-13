// Created by Zhiliang Zhou on 2020
// this is a simplified version of spconv
//
// original spconv was implemented by Yan Yan, https://github.com/traveller59/spconv
// -------------------------------------------------------------------
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
#include "spconv_cpp/pool_ops.h"
#include "spconv_cpp/maxpool.h"
#include "spconv_cpp/maxpool.cuh"
#include "tensorview/common.h"
// TV_ASSERT_INVALID_ARG


namespace spconv {

torch::Tensor indiceMaxPool(torch::Tensor features,
                            torch::Tensor indicePairs,
                            torch::Tensor indiceNum,
                            int64_t numAct) {

    auto device = features.device().type();
    auto kernelVolume = indiceNum.size(0);
    auto numInPlanes = features.size(1);
    auto indicePairNumCpu = indiceNum.to({torch::kCPU});
    auto options = torch::TensorOptions().dtype(features.dtype()).device(features.device());
    torch::Tensor output = torch::zeros({numAct, numInPlanes}, options);
//    double totalTime = 0;
    for (int i = 0; i < kernelVolume; ++i) {
        auto nHot = indicePairNumCpu.data_ptr<int>()[i];
        if (nHot <= 0) {
            continue;
        }
        // auto timer = spconv::CudaContextTimer<>();
        if (device == torch::kCPU) {
            maxpool_fwd_cpu(output, features, indicePairs[0][i], indicePairs[1][i], nHot);
        }
#ifdef TV_CUDA
        else if (device == torch::kCUDA) {
            maxpool_fwd_cuda(output, features, indicePairs[0][i], indicePairs[1][i], nHot);
        }
#endif
        else {
            TV_ASSERT_INVALID_ARG(false, "unknown device type");
        }
        // totalTime += timer.report() / 1000.0;
    }
    // std::cout << "maxpool forward time " << totalTime << std::endl;
    return output;
}

torch::Tensor indiceMaxPoolBackward(torch::Tensor features,
                                    torch::Tensor outFeatures,
                                    torch::Tensor outGrad,
                                    torch::Tensor indicePairs,
                                    torch::Tensor indiceNum) {
    auto device = features.device().type();
//    auto numInPlanes = features.size(1);
    auto indicePairNumCpu = indiceNum.to({torch::kCPU});
    auto options = torch::TensorOptions().dtype(features.dtype()).device(features.device());
    torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
    auto kernelVolume = indiceNum.size(0);
    for (int i = 0; i < kernelVolume; ++i) {
        auto nHot = indicePairNumCpu.data_ptr<int>()[i];
        if (nHot <= 0) {
            continue;
        }
        if (device == torch::kCPU) {
            maxpool_bwd_cpu(outFeatures,
                            features,
                            outGrad,
                            inputGrad,
                            indicePairs[0][i],
                            indicePairs[1][i],
                            nHot);
        }
#ifdef TV_CUDA
        else if (device == torch::kCUDA) {
            maxpool_bwd_cuda(outFeatures,
                             features,
                             outGrad,
                             inputGrad,
                             indicePairs[0][i],
                             indicePairs[1][i],
                             nHot);
        }
#endif
        else {
            TV_ASSERT_INVALID_ARG(false, "unknown device type");
        }
    }
    return inputGrad;
}

} // namespace spconv