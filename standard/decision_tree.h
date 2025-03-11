#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>

// The structure here represents a node in the decision tree
struct Node {
    int feature_index;
    double threshold;
    Node* left;
    Node* right;
    int value; // For leaf nodes
};

// Class that represents a Decision Tree for classification
class DecisionTree {
public:
    DecisionTree(int max_depth = 10);
    ~DecisionTree();
    void fit(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);
    int predict(const std::vector<double>& sample);
private:
    Node* root;
    int max_depth;

    Node* build_tree(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int depth);
    double calculate_gini(const std::vector<int>& labels);
    void delete_tree(Node* node);
    int most_common_label(const std::vector<int>& labels);

};

#endif
