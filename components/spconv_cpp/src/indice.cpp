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

#include "spconv_cpp/indice.h"

#include <ATen/Parallel.h>
// at::parallel_for()
#include "spconv_cpp/geometry.h"
// getValidOutPos()
// getValidOutPosTranspose()
#include "tensorview/torch_utils.h"
// tv::dispatch_torch()
#include "tensorview/tensorview.h"
// tv::rowArrayIdx
// tv::TensorView
// tv::SimpleVector
#include "tensorview/tensor.h"
// tv::dispatch_int()
#include "tsl/robin_map.h"
// tsl::robin_map

namespace spconv {

template <typename Index, typename IndexGrid, unsigned NDim>
Index getIndicePairsConv(tv::TensorView<Index> indicesIn,
                         tv::TensorView<Index> indicesOut,
                         tv::TensorView<IndexGrid> gridsOut,
                         tv::TensorView<Index> indicePairs,
                         tv::TensorView<Index> indiceNum,
                         const Index *kernelSize, const Index *stride,
                         const Index *padding, const Index *dilation,
                         const Index *outSpatialShape)
{
    // indicesOut: num_active * kernelVolume * (NDim + 1)
    Index numAct = 0;
    auto numActIn = indicesIn.dim(0);
    Index batchIdx = 0;
    Index spatialVolume = 1;
#pragma unroll
    for (unsigned int i = 0; i < NDim; ++i) {
        spatialVolume *= outSpatialShape[i];
    }
    Index kernelVolume = 1;
#pragma unroll
    for (unsigned int i = 0; i < NDim; ++i) {
        kernelVolume *= kernelSize[i];
    }
    Index numValidPoints = 0;
    std::vector<Index> validPoints_(kernelVolume * (NDim + 1));
    Index *validPoints = validPoints_.data();
    Index *pointPtr = nullptr;
    Index hashval;
    tsl::robin_map<Index, Index> hash;
    for (int j = 0; j < numActIn; ++j) {
        batchIdx = indicesIn(j, 0);
        numValidPoints = getValidOutPos<Index, NDim>(
                indicesIn.data() + j * (NDim + 1) + 1, kernelSize, stride, padding,
                dilation, outSpatialShape, validPoints);
        for (Index i = 0; i < numValidPoints; ++i) {
            pointPtr = validPoints + i * (NDim + 1);
            auto offset = pointPtr[NDim];
            auto index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape) +
                         spatialVolume * batchIdx;
            auto iter = hash.find(index);
            if (iter == hash.end()) {
                for (unsigned k = 1; k < NDim + 1; ++k) {
                    indicesOut(numAct, k) = pointPtr[k - 1];
                }
                indicesOut(numAct, 0) = batchIdx;
                hashval = numAct++;
                hash[index] = hashval;
            } else {
                hashval = iter->second;
            }
            // indicePairs: [K, 2, L]
            indicePairs(0, offset, indiceNum[offset]) = j;
            indicePairs(1, offset, indiceNum[offset]++) = hashval;
        }
    }
    return numAct;
}

template <typename Index, typename IndexGrid, unsigned NDim>
Index getIndicePairsDeConv(tv::TensorView<Index> indicesIn,
                           tv::TensorView<Index> indicesOut,
                           tv::TensorView<IndexGrid> gridsOut,
                           tv::TensorView<Index> indicePairs,
                           tv::TensorView<Index> indiceNum,
                           const Index *kernelSize, const Index *stride,
                           const Index *padding, const Index *dilation,
                           const Index *outSpatialShape) {
    Index numAct = 0;
    auto numActIn = indicesIn.dim(0);
    Index batchIdx = 0;
    Index spatialVolume = 1;
#pragma unroll
    for (unsigned int i = 0; i < NDim; ++i) {
        spatialVolume *= outSpatialShape[i];
    }
    Index kernelVolume = 1;
#pragma unroll
    for (unsigned int i = 0; i < NDim; ++i) {
        kernelVolume *= kernelSize[i];
    }
    Index numValidPoints = 0;
    std::vector<Index> validPoints_(kernelVolume * (NDim + 1));
    Index *validPoints = validPoints_.data();
    Index *pointPtr = nullptr;
    Index hashval;
    tsl::robin_map<Index, Index> hash;
    for (int j = 0; j < numActIn; ++j) {
        batchIdx = indicesIn(j, 0);
        numValidPoints = getValidOutPosTranspose<Index, NDim>(
                indicesIn.data() + j * (NDim + 1) + 1, kernelSize, stride, padding,
                dilation, outSpatialShape, validPoints);
        for (Index i = 0; i < numValidPoints; ++i) {
            pointPtr = validPoints + i * (NDim + 1);
            auto offset = pointPtr[NDim];
            auto index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape) +
                         spatialVolume * batchIdx;

            auto iter = hash.find(index);
            if (iter == hash.end()) {
                for (unsigned k = 1; k < NDim + 1; ++k) {
                    indicesOut(numAct, k) = pointPtr[k - 1];
                }
                indicesOut(numAct, 0) = batchIdx;
                hashval = numAct++;
                hash[index] = hashval;
            } else {
                hashval = iter->second;
            }
            // indicePairs: [K, 2, L]
            indicePairs(0, offset, indiceNum[offset]) = j;
            indicePairs(1, offset, indiceNum[offset]++) = hashval;
        }
    }
    return numAct;
}

template <typename Index, typename IndexGrid, unsigned NDim>
Index getIndicePairsSubM(tv::TensorView<Index> indicesIn,
                         tv::TensorView<IndexGrid> gridsOut,
                         tv::TensorView<Index> indicePairs,
                         tv::TensorView<Index> indiceNum,
                         const Index *const kernelSize,
                         const Index *const stride, const Index *const padding,
                         const Index *dilation,
                         const Index *const outSpatialShape) {
//    Index numAct = 0;
    auto numActIn = indicesIn.dim(0);
//    Index batchIdx = 0;
    Index spatialVolume = 1;
#pragma unroll
    for (unsigned int i = 0; i < NDim; ++i) {
        spatialVolume *= outSpatialShape[i];
    }
    Index kernelVolume = 1;
#pragma unroll
    for (unsigned int i = 0; i < NDim; ++i) {
        kernelVolume *= kernelSize[i];
    }
    tsl::robin_map<Index, Index> hash;
    for (int j = 0; j < numActIn; ++j) {
        Index index = 0;
        index = tv::rowArrayIdx<Index, NDim>(indicesIn.data() + j * (NDim + 1) + 1,
                                             outSpatialShape) +
                spatialVolume * indicesIn(j, 0);
        hash[index] = j;
    }

    at::parallel_for(0, numActIn, 0, [&](int64_t begin, int64_t end) {
        Index index = 0;
        Index numValidPoints = 0;
        std::vector<Index> validPoints_(kernelVolume * (NDim + 1));
        Index *validPoints = validPoints_.data();
        Index *pointPtr = nullptr;
        Index oldOffset = 0;
        for (int j = begin; j < end; ++j) {
            numValidPoints = getValidOutPos<Index, NDim>(
                    indicesIn.data() + j * (NDim + 1) + 1, kernelSize, stride, padding,
                    dilation, outSpatialShape, validPoints);
            for (Index i = 0; i < numValidPoints; ++i) {
                pointPtr = validPoints + i * (NDim + 1);
                auto offset = pointPtr[NDim];
                index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape) +
                        spatialVolume * indicesIn(j, 0);
                auto iter = hash.find(index);
                if (iter != hash.end()) {
#pragma omp atomic capture
                    oldOffset = indiceNum[offset]++;
                    indicePairs(0, offset, oldOffset) = j;
                    indicePairs(1, offset, oldOffset) = iter->second;
                }
            }
        }
    });
    return numActIn;
}


int create_conv_indice_pair_cpu(
        torch::Tensor indicesIn, torch::Tensor indicesOut, torch::Tensor gridsOut,
        torch::Tensor indicePairs, torch::Tensor indiceNum,
        std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
        std::vector<int64_t> padding, std::vector<int64_t> dilation,
        std::vector<int64_t> outSpatialShape, bool transpose, bool resetGrid,
        bool useHash) {
    auto ndim = outSpatialShape.size();
    auto numActIn = indicesIn.size(0);
//    int batchSize = gridsOut.size(0);
//    auto kernelVolume = indiceNum.size(0);
    if (numActIn == 0)
        return 0;
    // this block is a check-go wrapper with type deduce
    // go if indicesIn.scalar_type() is int32_t or int64_t
    tv::dispatch_torch<int32_t, int64_t>(indicesIn.scalar_type(), [&](auto V) {
        using Index = decltype(V);
        using IndexGrid = int32_t;
        // go if ndim is 2 or 3 or 4
        tv::dispatch_int<2, 3, 4>(ndim, [&](auto I) {
            constexpr int NDim = decltype(I)::value;
            tv::SimpleVector<Index, NDim> ks(kernelSize.begin(), kernelSize.end());
            tv::SimpleVector<Index, NDim> st(stride.begin(), stride.end());
            tv::SimpleVector<Index, NDim> pa(padding.begin(), padding.end());
            tv::SimpleVector<Index, NDim> di(dilation.begin(), dilation.end());
            tv::SimpleVector<Index, NDim> ou(outSpatialShape.begin(),
                                             outSpatialShape.end());
            if (transpose)
                numActIn = getIndicePairsDeConv<Index, IndexGrid, NDim>(
                        tv::torch2tv<Index>(indicesIn),
                        tv::torch2tv<Index>(indicesOut),
                        tv::torch2tv<IndexGrid>(gridsOut),
                        tv::torch2tv<Index>(indicePairs),
                        tv::torch2tv<Index>(indiceNum),
                        ks.data(), st.data(), pa.data(),
                        di.data(), ou.data());
            else
                numActIn = getIndicePairsConv<Index, IndexGrid, NDim>(
                        tv::torch2tv<Index>(indicesIn), tv::torch2tv<Index>(indicesOut),
                        tv::torch2tv<IndexGrid>(gridsOut), tv::torch2tv<Index>(indicePairs),
                        tv::torch2tv<Index>(indiceNum), ks.data(), st.data(), pa.data(),
                        di.data(), ou.data());
        });
    });
    return numActIn;
}

int create_submconv_indice_pair_cpu(
        torch::Tensor indicesIn, torch::Tensor gridsOut, torch::Tensor indicePairs,
        torch::Tensor indiceNum, std::vector<int64_t> kernelSize,
        std::vector<int64_t> stride, std::vector<int64_t> padding,
        std::vector<int64_t> dilation, std::vector<int64_t> outSpatialShape,
        bool transpose, bool resetGrid, bool useHash) {
    auto ndim = outSpatialShape.size();
    auto numActIn = indicesIn.size(0);
//    int batchSize = gridsOut.size(0);
//    auto kernelVolume = indiceNum.size(0);
    if (numActIn == 0)
        return 0;
    tv::dispatch_torch<int32_t, int64_t>(indicesIn.scalar_type(), [&](auto V) {
        using Index = decltype(V);
        using IndexGrid = int32_t;
        tv::dispatch_int<2, 3, 4>(ndim, [&](auto I) {
            constexpr int NDim = decltype(I)::value;
            tv::SimpleVector<Index, NDim> ks(kernelSize.begin(), kernelSize.end());
            tv::SimpleVector<Index, NDim> st(stride.begin(), stride.end());
            tv::SimpleVector<Index, NDim> pa(padding.begin(), padding.end());
            tv::SimpleVector<Index, NDim> di(dilation.begin(), dilation.end());
            tv::SimpleVector<Index, NDim> ou(outSpatialShape.begin(),
                                             outSpatialShape.end());
            numActIn = getIndicePairsSubM<Index, IndexGrid, NDim>(
                    tv::torch2tv<Index>(indicesIn), tv::torch2tv<IndexGrid>(gridsOut),
                    tv::torch2tv<Index>(indicePairs), tv::torch2tv<Index>(indiceNum),
                    ks.data(), st.data(), pa.data(), di.data(), ou.data());
        });
    });
    return numActIn;
}



}