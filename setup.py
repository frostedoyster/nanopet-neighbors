from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import subprocess


nanopet_neighbors_cpu = CppExtension(
    name="nanopet_neighbors_cpu",
    sources=["src/nanopet_neighbors.cc"],
)

nanopet_neighbors_cuda = CUDAExtension(
    name="nanopet_neighbors_cuda",
    sources=["src/nanopet_neighbors.cu"],
)

extensions = [nanopet_neighbors_cpu]
# try to find out if nvcc is available
try:
    nvcc_path = subprocess.check_output("which nvcc", shell=True).decode("utf-8").strip()
    extensions.append(nanopet_neighbors_cuda)
except:
    pass

setup(
    name="nanopet-neighbors",
    version="0.0.0",
    packages=["nanopet_neighbors"],
    package_dir={"nanopet_neighbors": "src"},
    install_requires=["torch"],
    ext_modules=extensions,
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    },
    package_data={
        'nanopet_neighbors': ['nanopet_neighbors_cpu.so', 'nanopet_neighbors_cuda.so'],
    },
    include_package_data=True,
)
