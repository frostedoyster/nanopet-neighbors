import torch
import pytest

from nanopet_neighbors import get_nef_indices, get_corresponding_edges
from reference_implementations import get_nef_indices as get_nef_indices_ref
from reference_implementations import get_corresponding_edges as get_corresponding_edges_ref


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_nef_indices(device):
    n_nodes = 1000
    n_edges_per_node = 30
    centers = torch.randint(0, 100, (1000,), device=device)

    result = get_nef_indices(centers, n_nodes, n_edges_per_node)
    result_ref = get_nef_indices_ref(centers, n_nodes, n_edges_per_node)

    for r, r_ref in zip(result, result_ref):
        assert torch.equal(r, r_ref)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_corresponding_edges(device):
    array = torch.randint(0, 5, (10, 2), device=device)
    array = torch.cat([array, array.flip(1)])

    result = get_corresponding_edges(array)
    result_ref = get_corresponding_edges_ref(array)

    print(result.device, result_ref.device)

    assert torch.equal(result, result_ref)
