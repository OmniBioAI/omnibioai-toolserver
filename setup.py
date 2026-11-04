"""
Cython build configuration for omnibioai-toolserver IP protection.
Usage: python setup.py build_ext --inplace
"""
import os

from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import find_packages, setup
from setuptools.extension import Extension

Options.annotate = False

EXTENSIONS = [
    "toolserver/tools/david_annotation.py",
    "toolserver/tools/enrichr_pathway.py",
    "toolserver/executor.py",
]


def make_extensions(paths):
    exts = []
    for p in paths:
        if not os.path.exists(p):
            print(f"WARNING: {p} not found, skipping")
            continue
        module = p.replace("/", ".").replace("\\", ".").removesuffix(".py")
        exts.append(Extension(module, [p]))
    return exts


setup(
    name="omnibioai-toolserver",
    packages=find_packages(exclude=["tests*"]),
    ext_modules=cythonize(
        make_extensions(EXTENSIONS),
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        nthreads=os.cpu_count() or 4,
    ),
    zip_safe=False,
)
