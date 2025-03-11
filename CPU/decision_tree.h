#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>

using namespace std;

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
    void fit(const vector<vector<double>>& data, const vector<int>& labels);
    int predict(const vector<double>& sample);
private:
    Node* root;
    int max_depth;

    Node* build_tree(const vector<vector<double>>& data, const vector<int>& labels, int depth);
    double calculate_gini(const vector<int>& labels);
    void delete_tree(Node* node);
    int most_common_label(const vector<int>& labels);

};

#endif