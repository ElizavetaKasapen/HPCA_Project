#ifndef DECISION_TREE_CUDA_V2_H
#define DECISION_TREE_CUDA_V2_H

#include <vector>

// Node structure for the Decision Tree
struct Node {
    int feature_index;   // Index of feature used for splitting
    double threshold;    // Threshold value for splitting
    Node* left;          // Pointer to the left subtree
    Node* right;         // Pointer to the right subtree
    int value;           // Class label (used for leaf nodes)
};

// Decision Tree class definition using CUDA for split evaluation.
class DecisionTree {
public:
    // Constructor with an optional maximum depth (default is 10)
    DecisionTree(int max_depth = 10);

    // Destructor to free allocated tree nodes
    ~DecisionTree();

    // Train the decision tree using the provided data and labels
    void fit(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);

    // Predict the class label for a given input sample
    int predict(const std::vector<double>& sample);

private:
    // Optional helper function to find the best split using GPU.
    // You can integrate GPU-based split evaluation directly in build_tree if preferred.
    void find_split_on_gpu(const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        int& best_feature,
        double& best_threshold);

    // Recursively builds the decision tree
    Node* build_tree(const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels,
        int depth);

    // Calculates the Gini impurity for a set of labels
    double calculate_gini(const std::vector<int>& labels);

    // Recursively frees the memory allocated for the tree nodes
    void delete_tree(Node* node);

    // Determines the most common label among a vector of labels
    int most_common_label(const std::vector<int>& labels);

    Node* root;     // Pointer to the root node of the decision tree
    int max_depth;  // Maximum allowed depth of the tree
};

#endif // DECISION_TREE_CUDA_V2_H
