import torch
import pytest

from nanopet_neighbors import get_nef_indices, get_corresponding_edges
from reference_implementations import get_nef_indices as get_nef_indices_ref
from reference_implementations import get_corresponding_edges as get_corresponding_edges_ref


torch.manual_seed(0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("torchscript", [False, True])
def test_get_nef_indices(device, torchscript):
    n_nodes = 1000
    n_edges_per_node = 30
    centers = torch.randint(0, 100, (1000,), device=device)

    if torchscript:
        get_nef_indices_to_use = torch.jit.script(get_nef_indices)
    else:
        get_nef_indices_to_use = get_nef_indices

    result = get_nef_indices_to_use(centers, n_nodes, n_edges_per_node)
    result_ref = get_nef_indices_ref(centers, n_nodes, n_edges_per_node)

    for r, r_ref in zip(result, result_ref):
        assert torch.equal(r, r_ref)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("torchscript", [False, True])
def test_get_corresponding_edges(device, torchscript):
    array = torch.randint(0, 5, (10, 2), device=device)
    shifts = torch.randint(0, 5, (10, 3), device=device)
    neighbor_list = torch.cat([
        torch.cat([array, array.flip(1)]),
        torch.cat([shifts, -shifts]),
    ], dim=1)

    if torchscript:
        get_corresponding_edges_to_use = torch.jit.script(get_corresponding_edges)
    else:
        get_corresponding_edges_to_use = get_corresponding_edges

    result = get_corresponding_edges_to_use(neighbor_list)
    result_ref = get_corresponding_edges_ref(neighbor_list)

    assert torch.equal(result, result_ref)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_corresponding_edges_error(device):
    array = torch.randint(0, 5, (10, 5), device=device)

    with pytest.raises(Exception):
        get_corresponding_edges(array)
        # This will actually fail with a CUDA assert, but we can't catch that
        # in a Python test...
