#include <torch/extension.h>
#include <cmath>


__global__ void get_nef_indices_kernel(
    const long* centers_ptr,
    long* edges_to_nef_ptr,
    long* nef_to_edges_neighbor_ptr,
    bool* nef_mask_ptr,
    int64_t n_edges,
    int64_t n_nodes,
    int64_t n_edges_per_node
) {
    int center = blockIdx.x * blockDim.x + threadIdx.x;
    int node_counter = 0;

    // parallelize over centers, and not edges, to avoid concurrent writes to edges_to_nef
    if (center < n_nodes) {
        for (int64_t i_edge = 0; i_edge < n_edges; i_edge++) {
            if (centers_ptr[i_edge] == center) {
                edges_to_nef_ptr[center * n_edges_per_node + node_counter] = i_edge;
                nef_mask_ptr[center * n_edges_per_node + node_counter] = true;
                nef_to_edges_neighbor_ptr[i_edge] = node_counter;
                node_counter++;
            }
        }
    }
}

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
    torch::Tensor nef_mask = torch::full(
        {n_nodes, n_edges_per_node}, 0, torch::TensorOptions().dtype(torch::kBool).device(centers.device())
    );

    long* centers_ptr = centers.data_ptr<long>();
    long* edges_to_nef_ptr = edges_to_nef.data_ptr<long>();
    long* nef_to_edges_neighbor_ptr = nef_to_edges_neighbor.data_ptr<long>();
    bool* nef_mask_ptr = nef_mask.data_ptr<bool>();

    // cudaPointerAttributes attributes;
    // cudaPointerGetAttributes(&attributes, centers_ptr);
    // int current_device;
    // cudaGetDevice(&current_device);
    // if (current_device != attributes.device) {
    //     cudaSetDevice(attributes.device);
    // }

    int threads_per_block = 256;
    int num_blocks = (n_nodes + threads_per_block - 1) / threads_per_block;
    if (n_nodes > 0) {
        get_nef_indices_kernel<<<num_blocks, threads_per_block>>>(
            centers_ptr,
            edges_to_nef_ptr,
            nef_to_edges_neighbor_ptr,
            nef_mask_ptr,
            n_edges,
            n_nodes,
            n_edges_per_node
        );
    }

    // cudaSetDevice(current_device);

    return {edges_to_nef, nef_to_edges_neighbor, nef_mask};
}


__global__ void find_corresponding_edges_kernel(
    const long* centers_ptr,
    const long* neighbors_ptr,
    const long* shift_x_ptr,
    const long* shift_y_ptr,
    const long* shift_z_ptr,
    long* inverse_indices_ptr,
    int64_t n_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_edges) {
        bool found = false;
        for (int64_t j = 0; j < n_edges; j++) {
            if (centers_ptr[i] == neighbors_ptr[j] && centers_ptr[j] == neighbors_ptr[i] && shift_x_ptr[i] == -shift_x_ptr[j] && shift_y_ptr[i] == -shift_y_ptr[j] && shift_z_ptr[i] == -shift_z_ptr[j]) {
                inverse_indices_ptr[i] = j;
                found = true;
                break;
            }
        }
        if (!found) {
            inverse_indices_ptr[i] = -1; // Use -1 to indicate no corresponding edge found
        }
    }
}

torch::Tensor get_corresponding_edges(
    torch::Tensor array
) {
    torch::Tensor centers = array.index({torch::indexing::Slice(), 0}).to(torch::kLong).contiguous();
    torch::Tensor neighbors = array.index({torch::indexing::Slice(), 1}).to(torch::kLong).contiguous();
    torch::Tensor shift_x = array.index({torch::indexing::Slice(), 2}).to(torch::kLong).contiguous();
    torch::Tensor shift_y = array.index({torch::indexing::Slice(), 3}).to(torch::kLong).contiguous();
    torch::Tensor shift_z = array.index({torch::indexing::Slice(), 4}).to(torch::kLong).contiguous();

    long* centers_ptr = centers.data_ptr<long>();
    long* neighbors_ptr = neighbors.data_ptr<long>();
    long* shift_x_ptr = shift_x.data_ptr<long>();
    long* shift_y_ptr = shift_y.data_ptr<long>();
    long* shift_z_ptr = shift_z.data_ptr<long>();

    int64_t n_edges = centers.size(0);

    torch::Tensor inverse_indices = torch::empty(
        {n_edges}, torch::TensorOptions().dtype(torch::kLong).device(centers.device())
    );
    long* inverse_indices_ptr = inverse_indices.data_ptr<long>();

    // cudaPointerAttributes attributes;
    // cudaPointerGetAttributes(&attributes, centers_ptr);
    // int current_device;
    // cudaGetDevice(&current_device);
    // if (current_device != attributes.device) {
    //     cudaSetDevice(attributes.device);
    // }

    int threads_per_block = 256;
    int num_blocks = (n_edges + threads_per_block - 1) / threads_per_block;
    if (n_edges > 0) {
        find_corresponding_edges_kernel<<<num_blocks, threads_per_block>>>(
            centers_ptr,
            neighbors_ptr,
            shift_x_ptr,
            shift_y_ptr,
            shift_z_ptr,
            inverse_indices_ptr,
            n_edges
        );
    }

    // cudaSetDevice(current_device);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }

    if (torch::any(inverse_indices == -1).item<bool>()) {
        throw std::runtime_error("Some edges do not have corresponding edges");
    }

    return inverse_indices;
}


TORCH_LIBRARY(nanopet_neighbors_cuda, m) {
    m.def(
        "get_nef_indices",
         &get_nef_indices
    );
    m.def(
        "get_corresponding_edges",
         &get_corresponding_edges
    );
}
