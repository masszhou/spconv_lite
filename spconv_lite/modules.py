# Created by Zhiliang Zhou on 2020
# changes:
# 1. remove fused() from SparseSequential
#
# original implemented by Yan Yan, https://github.com/traveller59/spconv
# ----------------------------------------------------------------
# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from collections import OrderedDict
from torch import nn
from spconv_lite.sparse_tensor import SparseConvTensor


def is_spconv_module(module_type):
    spconv_modules = (SparseModule, )
    return isinstance(module_type, spconv_modules)


class SparseModule(nn.Module):
    """ place holder, all module subclass from this will take sptensor in SparseSequential.
    """
    pass


class SparseSequential(SparseModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = SparseSequential(
                  SparseConv2d(1,20,5),
                  nn.ReLU(),
                  SparseConv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = SparseSequential(OrderedDict([
                  ('conv1', SparseConv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', SparseConv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = SparseSequential(
                  conv1=SparseConv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=SparseConv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """
    def __init__(self, *args, **kwargs):
        super(SparseSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)
        self._sparity_dict = {}

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    @property
    def sparity_dict(self):
        return self._sparity_dict

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            if is_spconv_module(module):  # use SpConvTensor as input
                if isinstance(input, list):
                    input = module(input)
                else:
                    assert isinstance(input, SparseConvTensor)
                    self._sparity_dict[k] = input.sparity
                    input = module(input)
            else:
                if isinstance(input, SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input.features = module(input.features)
                else:
                    input = module(input)
        return input


class ToDense(SparseModule):
    """convert SparseConvTensor to NCHW dense tensor.
    """
    def forward(self, x: SparseConvTensor):
        return x.dense()


class RemoveGrid(SparseModule):
    """remove pre-allocated grid buffer.
    """
    def forward(self, x: SparseConvTensor):
        x.grid = None
        return x