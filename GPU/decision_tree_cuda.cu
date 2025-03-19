#include "decision_tree.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_CLASSES 3 

// CUDA kernel: each thread evaluates one candidate split
__global__ void evaluate_candidates(const double* d_data, const int* d_labels,
    int num_samples, int num_features,
    double* d_results, int* d_candidate_feature,
    double* d_candidate_threshold) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_candidates = num_features * num_samples;
    if (idx >= total_candidates) return;

    // Determine candidate: each candidate corresponds to (feature_index, sample_index)
    int feature_index = idx / num_samples;
    int sample_index = idx % num_samples;
    double threshold = d_data[sample_index * num_features + feature_index];

    int left_total = 0;
    int right_total = 0;
    int left_counts[MAX_CLASSES] = { 0 };
    int right_counts[MAX_CLASSES] = { 0 };

    // Evaluate the split for all samples using the given feature
    for (int i = 0; i < num_samples; i++) {
        double value = d_data[i * num_features + feature_index];
        int label = d_labels[i];
        if (value <= threshold) {
            left_total++;
            if (label < MAX_CLASSES)
                left_counts[label]++;
        }
        else {
            right_total++;
            if (label < MAX_CLASSES)
                right_counts[label]++;
        }
    }

    // Compute Gini for left branch
    double gini_left = 1.0;
    if (left_total > 0) {
        for (int c = 0; c < MAX_CLASSES; c++) {
            double p = static_cast<double>(left_counts[c]) / left_total;
            gini_left -= p * p;
        }
    }

    // Compute Gini for right branch
    double gini_right = 1.0;
    if (right_total > 0) {
        for (int c = 0; c < MAX_CLASSES; c++) {
            double p = static_cast<double>(right_counts[c]) / right_total;
            gini_right -= p * p;
        }
    }

    // Compute weighted Gini 
    double weighted_gini = 0.0;
    if (num_samples > 0) {
        weighted_gini = (left_total * gini_left + right_total * gini_right) / num_samples;
    }

    // Write results to global memory
    d_results[idx] = weighted_gini;
    d_candidate_feature[idx] = feature_index;
    d_candidate_threshold[idx] = threshold;
}


DecisionTree::DecisionTree(int max_depth) : max_depth(max_depth), root(nullptr) {}

DecisionTree::~DecisionTree() {
    delete_tree(root);
}

// Recursively delete all nodes in the tree
void DecisionTree::delete_tree(Node* node) {
    if (node) {
        delete_tree(node->left);
        delete_tree(node->right);
        delete node;
    }
}

// Finds the most common class label (used for leaf nodes)
int DecisionTree::most_common_label(const std::vector<int>& labels) {
    std::map<int, int> label_count;
    for (int label : labels) {
        label_count[label]++;
    }
    return std::max_element(label_count.begin(), label_count.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; })->first;
}


// Recursively builds the decision tree using the dataset and CUDA for candidate evaluation
Node* DecisionTree::build_tree(const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels, int depth) {
    int num_samples = data.size();
    int num_features = data[0].size();

    //std::cout << "Depth: " << depth << ", Samples: " << num_samples << std::endl;

    // Check stopping conditions
    if (depth >= max_depth || num_samples <= 2 ||
        std::all_of(labels.begin(), labels.end(), [&](int v) { return v == labels[0]; })) {
        Node* leaf = new Node();
        leaf->value = most_common_label(labels);
        leaf->left = nullptr;
        leaf->right = nullptr;
        //std::cout << "Created leaf node with value: " << leaf->value << std::endl;
        return leaf;
    }

    // Flatten 2D data into a contiguous 1D array for CUDA
    std::vector<double> flat_data(num_samples * num_features);
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            flat_data[i * num_features + j] = data[i][j];
        }
    }

    // Allocate device memory
    double* d_data, * d_results, * d_candidate_threshold;
    int* d_labels, * d_candidate_feature;
    size_t data_size = sizeof(double) * flat_data.size();
    size_t labels_size = sizeof(int) * num_samples;
    int total_candidates = num_features * num_samples;
    size_t candidates_size = sizeof(double) * total_candidates;
    size_t candidate_feature_size = sizeof(int) * total_candidates;

    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_labels, labels_size);
    cudaMalloc(&d_results, candidates_size);
    cudaMalloc(&d_candidate_feature, candidate_feature_size);
    cudaMalloc(&d_candidate_threshold, candidates_size);

    cudaMemcpy(d_data, flat_data.data(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), labels_size, cudaMemcpyHostToDevice);

    // Launch kernel: one thread per candidate split
    int blockSize = 256;
    int numBlocks = (total_candidates + blockSize - 1) / blockSize;
    evaluate_candidates << <numBlocks, blockSize >> > (d_data, d_labels, num_samples,
        num_features,
        d_results, d_candidate_feature,
        d_candidate_threshold);
    cudaDeviceSynchronize();

    // Copy candidate evaluation results back to host
    std::vector<double> h_results(total_candidates);
    std::vector<int> h_candidate_feature(total_candidates);
    std::vector<double> h_candidate_threshold(total_candidates);
    cudaMemcpy(h_results.data(), d_results, candidates_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_candidate_feature.data(), d_candidate_feature, candidate_feature_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_candidate_threshold.data(), d_candidate_threshold, candidates_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_labels);
    cudaFree(d_results);
    cudaFree(d_candidate_feature);
    cudaFree(d_candidate_threshold);

    // Find the best candidate split on the host
    double best_gini = 1.0;
    int best_feature = -1;
    double best_threshold = 0.0;
    for (int i = 0; i < total_candidates; i++) {
        if (h_results[i] < best_gini) {
            best_gini = h_results[i];
            best_feature = h_candidate_feature[i];
            best_threshold = h_candidate_threshold[i];
        }
    }

    // If no valid split was found, create a leaf node
    if (best_feature == -1) {
        Node* leaf = new Node();
        leaf->value = most_common_label(labels);
        leaf->left = nullptr;
        leaf->right = nullptr;
       // std::cout << "Created fallback leaf node with value: " << leaf->value << std::endl;
        return leaf;
    }

    //std::cout << "Best feature: " << best_feature << ", Best threshold: " << best_threshold << std::endl;

    // Split the data into left and right branches based on the best split
    std::vector<std::vector<double>> left_data, right_data;
    std::vector<int> left_labels, right_labels;
    for (int i = 0; i < num_samples; i++) {
        if (data[i][best_feature] <= best_threshold) {
            left_data.push_back(data[i]);
            left_labels.push_back(labels[i]);
        }
        else {
            right_data.push_back(data[i]);
            right_labels.push_back(labels[i]);
        }
    }

    // Recursively build the tree
    Node* node = new Node();
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = build_tree(left_data, left_labels, depth + 1);
    node->right = build_tree(right_data, right_labels, depth + 1);

    return node;
}

void DecisionTree::fit(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    std::cout << "Starting to build the decision tree..." << std::endl;
    root = build_tree(data, labels, 0);
    std::cout << "Decision tree built successfully." << std::endl;
}


int DecisionTree::predict(const std::vector<double>& sample) {
    Node* node = root;
    while (node->left || node->right) {
        if (sample[node->feature_index] <= node->threshold) {
            node = node->left;
        }
        else {
            node = node->right;
        }
    }
    return node->value;
}
