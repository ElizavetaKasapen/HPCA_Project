#include "decision_tree.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <map>

DecisionTree::DecisionTree(int max_depth) : max_depth(max_depth), root(nullptr) {}

DecisionTree::~DecisionTree() {
    delete_tree(root);
}

void DecisionTree::delete_tree(Node* node) {
    if (node) {
        delete_tree(node->left);
        delete_tree(node->right);
        delete node;
    }
}

int DecisionTree::most_common_label(const std::vector<int>& labels) {
    std::map<int, int> label_count;

    for (int label : labels) {
        label_count[label]++;
    }

    return std::max_element(label_count.begin(), label_count.end(),
                            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
}

double DecisionTree::calculate_gini(const std::vector<int>& labels) {
    if (labels.empty()) return 0.0;

    std::map<int, int> label_count;
    for (int label : labels) {
        label_count[label]++;
    }

    double gini = 1.0;
    for (const auto& pair : label_count) {
        double probability = static_cast<double>(pair.second) / labels.size();
        gini -= probability * probability;
    }

    return gini;
}

// Recursive function to build a decision tree (sequential version)
Node* DecisionTree::build_tree(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int depth) {
    int num_samples = data.size();
    int num_features = data[0].size();

    //std::cout << "Depth: " << depth << ", Samples: " << num_samples << std::endl;

    if (depth >= max_depth || num_samples <= 2 || std::all_of(labels.begin(), labels.end(), [&](int v) { return v == labels[0]; })) {
        Node* leaf = new Node();
        leaf->value = most_common_label(labels);
        leaf->left = nullptr;
        leaf->right = nullptr;
        //std::cout << "Created leaf node with value: " << leaf->value << std::endl;
        return leaf;
    }

    double best_gini = 1.0;
    int best_feature = -1;
    double best_threshold = 0.0;

    for (int feature_index = 0; feature_index < num_features; feature_index++) {
        for (int sample_index = 0; sample_index < num_samples; sample_index++) {
            double threshold = data[sample_index][feature_index];

            std::vector<int> left_labels, right_labels;
            for (int i = 0; i < num_samples; i++) {
                if (data[i][feature_index] <= threshold) {
                    left_labels.push_back(labels[i]);
                } else {
                    right_labels.push_back(labels[i]);
                }
            }

            if (left_labels.empty() || right_labels.empty()) continue;

            double gini_left = calculate_gini(left_labels);
            double gini_right = calculate_gini(right_labels);
            double weighted_gini = (left_labels.size() * gini_left + right_labels.size() * gini_right) / num_samples;

            if (weighted_gini < best_gini) {
                best_gini = weighted_gini;
                best_feature = feature_index;
                best_threshold = threshold;
            }
        }
    }

    if (best_feature == -1) {
        Node* leaf = new Node();
        leaf->value = most_common_label(labels);
        leaf->left = nullptr;
        leaf->right = nullptr;
        //std::cout << "Created fallback leaf node with value: " << leaf->value << std::endl;
        return leaf;
    }

    //std::cout << "Best feature: " << best_feature << ", Best threshold: " << best_threshold << std::endl;

    Node* node = new Node();
    node->feature_index = best_feature;
    node->threshold = best_threshold;

    std::vector<std::vector<double>> left_data, right_data;
    std::vector<int> left_labels, right_labels;
    for (int i = 0; i < num_samples; i++) {
        if (data[i][best_feature] <= best_threshold) {
            left_data.push_back(data[i]);
            left_labels.push_back(labels[i]);
        } else {
            right_data.push_back(data[i]);
            right_labels.push_back(labels[i]);
        }
    }

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
        } else {
            node = node->right;
        }
    }
    return node->value;
}
