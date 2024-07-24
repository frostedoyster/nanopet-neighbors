import torch


def get_nef_indices(centers, n_nodes: int, n_edges_per_node: int):
    n_edges = len(centers)
    edges_to_nef = torch.zeros(
        (n_nodes, n_edges_per_node), dtype=torch.long, device=centers.device
    )
    nef_to_edges_neighbor = torch.empty(
        (n_edges,), dtype=torch.long, device=centers.device
    )
    node_counter = torch.zeros((n_nodes,), dtype=torch.long, device=centers.device)
    nef_mask = torch.full(
        (n_nodes, n_edges_per_node), 0, dtype=torch.bool, device=centers.device
    )
    for i in range(n_edges):
        center = centers[i]
        edges_to_nef[center, node_counter[center]] = i
        nef_mask[center, node_counter[center]] = True
        nef_to_edges_neighbor[i] = node_counter[center]
        node_counter[center] += 1
    return edges_to_nef, nef_to_edges_neighbor, nef_mask

def get_corresponding_edges(neighbor_list):
    array = neighbor_list[:, :2]
    shifts = neighbor_list[:, 2:]
    n_edges = len(array)
    neighbor_list_corresponding = torch.concatenate([array.flip(1), -shifts], dim=1)
    inverse_indices = torch.empty((n_edges,), dtype=torch.long, device=array.device)
    for i in range(n_edges):
        inverse_indices[i] = torch.nonzero(
            torch.all(neighbor_list_corresponding == neighbor_list[i], dim=1)
        )[0][0]
    return inverse_indices
