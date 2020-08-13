import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion
from pathlib import Path

import torch
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

LIBTORCH_ROOT = str(Path(torch.__file__).parent)
print(f"LIBTORCH_ROOT:{LIBTORCH_ROOT}")

SPCONV_FORCE_BUILD_CUDA = os.getenv("SPCONV_FORCE_BUILD_CUDA")

PYTHON_VERSION = "{}.{}".format(sys.version_info.major, sys.version_info.minor)

remove_plus = torch.__version__.find("+")
PYTORCH_VERSION = torch.__version__
if remove_plus != -1:
    PYTORCH_VERSION = torch.__version__[:remove_plus]
PYTORCH_VERSION = list(map(int, PYTORCH_VERSION.split(".")))
PYTORCH_VERSION_NUMBER = PYTORCH_VERSION[0] * 10000 + PYTORCH_VERSION[1] * 100 + PYTORCH_VERSION[2]


class CMakeExtension(Extension):
    """
    class distutils.core.Extension

    name: the full name of the extension, including any packages — ie. not a filename or pathname, but Python dotted name
    library_dirs: list of directories to search for C/C++ libraries at link time
    libraries: list of library names (not filenames or paths) to link against
    """
    def __init__(self, name, sourcedir='', library_dirs=[]):
       # don't invoke the original build_ext for this special extension
       Extension.__init__(self, name, sources=[], library_dirs=library_dirs)
       self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_PREFIX_PATH={}'.format(LIBTORCH_ROOT),
                      '-DPYBIND11_PYTHON_VERSION={}'.format(PYTHON_VERSION),
                      '-DPYTORCH_VERSION={}'.format(PYTORCH_VERSION_NUMBER),
                      '-DPYTHON_EXECUTABLE={}'.format(sys.executable)
                      ]

        cuda_flags = ["\"--expt-relaxed-constexpr\""]
        # must add following flags to use at::Half
        # but will remove raw half operators.
        cuda_flags += ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__"]
        cmake_args += ['-DCMAKE_CUDA_FLAGS=' + " ".join(cuda_flags)]

        cfg = 'Debug' if self.debug else 'Release'
        assert cfg == "Release", "pytorch ops don't support debug build."
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir]  # copy .so to package root
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print("|||||CMAKE ARGS|||||", cmake_args)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


# usage: python setup.py bdist_wheel
setup(
    name='spconv-lite',
    version='1.0.0',
    setup_requires = ['torch>=1.3.0'],
    packages=["spconv_lite", "spconv_lite.utils"],  # copy folder .spconv_lite to package root
    ext_modules=[CMakeExtension('spconv_lite._cpp_fn', library_dirs=[])],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,

    # metadata for upload to PyPI
    author='Zhiliang Zhou',
    author_email='zhouzhiliang@gmail.com',
    description='lite version of spatial sparse convolution for pytorch',
    long_description='original works and spconv implementation are written by Yan Yan, https://github.com/traveller59/spconv',
)
