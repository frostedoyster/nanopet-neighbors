#include <torch/extension.h>
#include <cmath>


std::vector<torch::Tensor> get_nef_indices(
    torch::Tensor centers,
    int64_t n_nodes,
    int64_t n_edges_per_node
) {
    centers = centers.to(torch::kLong).contiguous();

    int64_t n_edges = centers.size(0);
    torch::Tensor edges_to_nef = torch::zeros(
        {n_nodes, n_edges_per_node}, torch::TensorOptions().dtype(torch::kLong).device(centers.device())
    );
    torch::Tensor nef_to_edges_neighbor = torch::empty(
        {n_edges}, torch::TensorOptions().dtype(torch::kLong).device(centers.device())
    );
    std::vector<long> node_counter(n_nodes, 0);
    torch::Tensor nef_mask = torch::full(
        {n_nodes, n_edges_per_node}, 0, torch::TensorOptions().dtype(torch::kBool).device(centers.device())
    );

    long* centers_ptr = centers.data_ptr<long>();
    long* edges_to_nef_ptr = edges_to_nef.data_ptr<long>();
    long* nef_to_edges_neighbor_ptr = nef_to_edges_neighbor.data_ptr<long>();
    bool* nef_mask_ptr = nef_mask.data_ptr<bool>();

    for (int64_t i = 0; i < n_edges; i++) {
        long center = centers_ptr[i];
        edges_to_nef_ptr[center * n_edges_per_node + node_counter[center]] = i;
        nef_mask_ptr[center * n_edges_per_node + node_counter[center]] = true;
        nef_to_edges_neighbor_ptr[i] = node_counter[center];
        node_counter[center] += 1;
    }

    return {edges_to_nef, nef_to_edges_neighbor, nef_mask};
}


torch::Tensor get_corresponding_edges(
    torch::Tensor array
) {
    torch::Tensor centers = array.index({torch::indexing::Slice(), 0}).to(torch::kLong).contiguous();
    torch::Tensor neighbors = array.index({torch::indexing::Slice(), 1}).to(torch::kLong).contiguous();

    long* centers_ptr = centers.data_ptr<long>();
    long* neighbors_ptr = neighbors.data_ptr<long>();

    int64_t n_edges = centers.size(0);

    torch::Tensor inverse_indices = torch::empty(
        {n_edges}, torch::TensorOptions().dtype(torch::kLong).device(centers.device())
    );
    long* inverse_indices_ptr = inverse_indices.data_ptr<long>();

    for (int64_t i = 0; i < n_edges; i++) {
        for (int64_t j = 0; j < n_edges; j++) {
            if (centers_ptr[i] == neighbors_ptr[j] && centers_ptr[j] == neighbors_ptr[i]) {
                inverse_indices_ptr[i] = j;
                break;
            }
            if (j == n_edges - 1) throw std::runtime_error("No corresponding edge found");
        }
    }

    return inverse_indices;
}


TORCH_LIBRARY(nanopet_neighbors_cpu, m) {
    m.def(
        "get_nef_indices",
         &get_nef_indices
    );
    m.def(
        "get_corresponding_edges",
         &get_corresponding_edges
    );
}
