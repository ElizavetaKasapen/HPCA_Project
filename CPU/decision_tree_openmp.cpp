#include "decision_tree.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <map>


using namespace std;
// Constructor: It initializes the tree with a given maximum depth and sets root to nullptr
DecisionTree::DecisionTree(int max_depth) : max_depth(max_depth), root(nullptr) {}

// Destructor: It frees memory allocated for the tree nodes
DecisionTree::~DecisionTree() {
    delete_tree(root);
}

// A recursive function to delete all nodes in the tree
void DecisionTree::delete_tree(Node* node) {
    if (node) {
        delete_tree(node->left);
        delete_tree(node->right);
        delete node;
    }
}

// Function that finds the most common class label in a dataset which we use later for leaf nodes
int DecisionTree::most_common_label(const vector<int>& labels) {
    map<int, int> label_count;

    // Counting the occurrences of each label
    for (int label : labels) {
        label_count[label]++;
    }
    // Here we return the label with the highest frequency
    return max_element(label_count.begin(), label_count.end(), 
                            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
}

// Function that calculates the Gini impurity, that is a measure of how mixed the class labels are in our dataset
double DecisionTree::calculate_gini(const vector<int>& labels) {
    if (labels.empty()) return 0.0;

    map<int, int> label_count;
    for (int label : labels) {
        label_count[label]++;
    }

    double gini = 1.0;
    for (const auto& pair : label_count) {
        double probability = static_cast<double>(pair.second) / labels.size();
        gini -= probability * probability; // Gini impurity formula
    }

    return gini;
}

// Recursive function that builds a decision tree using the iris dataset
Node* DecisionTree::build_tree(const vector<vector<double>>& data, const vector<int>& labels, int depth) {
    int num_samples = data.size();
    int num_features = data[0].size();

    cout << "Depth: " << depth << ", Samples: " << num_samples << endl;

    // Implementation of stopping conditions:
    // - Maximum depth reached
    // - Too few samples to split
    // - All labels are the same (pure leaf)
    if (depth >= max_depth || num_samples <= 2 || all_of(labels.begin(), labels.end(), [&](int v) { return v == labels[0]; })) {
        Node* leaf = new Node();
        leaf->value = most_common_label(labels);  // Corrected leaf assignment
        leaf->left = nullptr;
        leaf->right = nullptr;
        cout << "Created leaf node with value: " << leaf->value << endl;
        return leaf;
    }

    double best_gini = 1.0;
    int best_feature = -1;
    double best_threshold = 0.0;

    // Usage of OpenMP parallelization for feature and threshold selection
    #pragma omp parallel
    {
        int local_best_feature = -1;
        double local_best_threshold = 0.0;
        double local_best_gini = 1.0;

        // Iteration over each feature in order to find the best split
        #pragma omp for nowait
        for (int feature_index = 0; feature_index < num_features; feature_index++) {
            for (int sample_index = 0; sample_index < num_samples; sample_index++) {
                double threshold = data[sample_index][feature_index];

                vector<int> left_labels, right_labels;
                for (int i = 0; i < num_samples; i++) {
                    if (data[i][feature_index] <= threshold) {
                        left_labels.push_back(labels[i]);
                    } else {
                        right_labels.push_back(labels[i]);
                    }
                }

                if (left_labels.empty() || right_labels.empty()) continue;

                // Calculation of weighted Gini impurity for the split
                double gini_left = calculate_gini(left_labels);
                double gini_right = calculate_gini(right_labels);
                double weighted_gini = (left_labels.size() * gini_left + right_labels.size() * gini_right) / num_samples;

                if (weighted_gini < local_best_gini) {
                    local_best_gini = weighted_gini;
                    local_best_feature = feature_index;
                    local_best_threshold = threshold;
                }
            }
        }

        // Here we update global best split
        #pragma omp critical
        {
            if (local_best_gini < best_gini) {
                best_gini = local_best_gini;
                best_feature = local_best_feature;
                best_threshold = local_best_threshold;
            }
        }
    }

    // In case no valid split is found, a leaf is returned 
    if (best_feature == -1) {
        Node* leaf = new Node();
        leaf->value = most_common_label(labels);
        leaf->left = nullptr;
        leaf->right = nullptr;
        cout << "Created fallback leaf node with value: " << leaf->value << endl;
        return leaf;
    }

    cout << "Best feature: " << best_feature << ", Best threshold: " << best_threshold << endl;

    // Creation of a new decision node
    Node* node = new Node();
    node->feature_index = best_feature;
    node->threshold = best_threshold;

    // Splitting of data into left and right branches
    vector<vector<double>> left_data, right_data;
    vector<int> left_labels, right_labels;
    for (int i = 0; i < num_samples; i++) {
        if (data[i][best_feature] <= best_threshold) {
            left_data.push_back(data[i]);
            left_labels.push_back(labels[i]);
        } else {
            right_data.push_back(data[i]);
            right_labels.push_back(labels[i]);
        }
    }

    // Recursively building the left and right subtrees
    node->left = build_tree(left_data, left_labels, depth + 1);
    node->right = build_tree(right_data, right_labels, depth + 1);

    return node;
}

// Function that trains the decision tree using the iris dataset
void DecisionTree::fit(const vector<vector<double>>& data, const vector<int>& labels) {
    cout << "Starting to build the decision tree..." << endl;
    root = build_tree(data, labels, 0);
    cout << "Decision tree built successfully." << endl;
}

// Function that predicts the class label for a given input sample
int DecisionTree::predict(const vector<double>& sample) {
    Node* node = root;

    // Traversing the tree based on feature thresholds
    while (node->left || node->right) {
        if (sample[node->feature_index] <= node->threshold) {
            node = node->left;
        } else {
            node = node->right;
        }
    }
    return node->value;
}