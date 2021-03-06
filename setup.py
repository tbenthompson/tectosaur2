import os
import sys
from os import path

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

debug = os.environ.get("TCT_DEBUG", "0") == "1"
print("Debug flag: ", debug)

if sys.platform == "win32":
    # debug currently not implemented on windows. easy to do...
    extra_compile_args = ["/openmp", "/O2"]
    extra_link_args = ["/openmp"]
else:
    extra_compile_args = [
        "-fopenmp",
        "--std=c++17",
        "-Wno-unreachable-code",
        "-Wno-sign-compare",
    ]
    if debug:
        extra_compile_args.extend(["-g", "-Og"])
    else:
        extra_compile_args.extend(["-O3", "-ffast-math"])

    extra_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        name="tectosaur2._ext",
        sources=["tectosaur2/_ext.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        depends=["tectosaur2/adaptive.hpp", "tectosaur2/direct_kernels.hpp"],
        language="c++",
    ),
    Extension(
        name="tectosaur2.hmatrix.aca_ext",
        sources=["tectosaur2/hmatrix/aca_ext.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        depends=["tectosaur2/direct_kernels.hpp"],
        language="c++",
    ),
]

Options.warning_errors = True

setup(
    name="tectosaur2",
    use_scm_version={"version_scheme": "post-release"},
    setup_requires=["setuptools_scm"],
    description="Boundary integral tools and tutorials, focused on earthquakes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tbenthompson/tectosaur2",
    author="T. Ben Thompson",
    author_email="t.ben.thompson@gmail.com",
    license="MIT",
    classifiers=[  # Optional
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    install_requires=[],
    entry_points=None,
    ext_modules=cythonize(ext_modules, annotate=False),
    zip_safe=False,
)
