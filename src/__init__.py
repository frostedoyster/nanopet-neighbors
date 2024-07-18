import torch
from distutils.sysconfig import get_python_lib
import os


try:
    site_packages_directory = get_python_lib()
    lib_path = os.path.join(site_packages_directory, "nanopet_neighbors.so")
    torch.ops.load_library(str(lib_path))
except Exception as e:
    print(f"Failed to load nanopet-neighbors: {e}")


def get_nef_indices(centers, n_nodes: int, n_edges_per_node: int):
    original_dtype = centers.dtype
    original_device = centers.device

    centers = centers.to("cpu", torch.long)
    edges_to_nef, nef_to_edges_neighbor, nef_mask = torch.ops.nanopet_neighbors.get_nef_indices(
        centers, n_nodes, n_edges_per_node
    )

    return (
        edges_to_nef.to(original_device, original_dtype),
        nef_to_edges_neighbor.to(original_device, original_dtype),
        nef_mask.to(original_device),
    )


def get_corresponding_edges(array):
    original_dtype = array.dtype
    original_device = array.device

    array = array.to("cpu", torch.long)
    inverse_indices = torch.ops.nanopet_neighbors.get_corresponding_edges(array)

    return inverse_indices.to(original_device, original_dtype)
