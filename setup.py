from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, include_paths, library_paths

include_dirs = include_paths()
library_dirs = library_paths()
libraries = ['c10', 'torch', 'torch_cpu']

nanopet_neighbors = Extension(
    name="nanopet_neighbors",
    sources=["src/nanopet_neighbors.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    language='c++',
)

setup(
    name="nanopet-neighbors",
    version="0.0.0",
    packages=["nanopet_neighbors"],
    package_dir={"nanopet_neighbors": "src"},
    install_requires=["torch"],
    ext_modules=[nanopet_neighbors],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    },
    package_data={
        'nanopet_neighbors': ['nanopet_neighbors.so'],
    },
    include_package_data=True,
)
