import torch
from distutils.sysconfig import get_python_lib
import os


try:
    site_packages_directory = get_python_lib()
    lib_path = os.path.join(site_packages_directory, "nanopet_neighbors_cpu.so")
    torch.ops.load_library(str(lib_path))
except Exception as e:
    print(f"Failed to load nanopet-neighbors: {e}")

try:
    site_packages_directory = get_python_lib()
    lib_path = os.path.join(site_packages_directory, "nanopet_neighbors_cuda.so")
    torch.ops.load_library(str(lib_path))
except:
    pass


def get_nef_indices(centers, n_nodes: int, n_edges_per_node: int):
    original_dtype = centers.dtype
    centers = centers.to(torch.long)

    device = centers.device
    if device.type == "cpu":
        edges_to_nef, nef_to_edges_neighbor, nef_mask = torch.ops.nanopet_neighbors_cpu.get_nef_indices(
            centers, n_nodes, n_edges_per_node
        )
    elif device.type == "cuda":
        edges_to_nef, nef_to_edges_neighbor, nef_mask = torch.ops.nanopet_neighbors_cuda.get_nef_indices(
            centers, n_nodes, n_edges_per_node
        )
    else:
        raise ValueError(f"Unsupported device: {device}")

    return edges_to_nef.to(original_dtype), nef_to_edges_neighbor.to(original_dtype), nef_mask


def get_corresponding_edges(array):
    original_dtype = array.dtype
    array = array.to(torch.long)

    device = array.device
    if device.type == "cpu":
        inverse_indices = torch.ops.nanopet_neighbors_cpu.get_corresponding_edges(array)
    elif device.type == "cuda":
        inverse_indices = torch.ops.nanopet_neighbors_cuda.get_corresponding_edges(array)
    else:
        raise ValueError(f"Unsupported device: {device}")

    return inverse_indices.to(original_dtype)
