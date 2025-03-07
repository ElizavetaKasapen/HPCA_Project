#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <chrono>

class TreeNode {
public:
    int featureIndex;
    double threshold;
    TreeNode* left;
    TreeNode* right;
    int label;

    TreeNode() : featureIndex(-1), threshold(0), left(nullptr), right(nullptr), label(-1) {}
};

class DecisionTree {
public:
    TreeNode* root;
    int maxDepth;

    DecisionTree(int depth) : root(nullptr), maxDepth(depth) {}

    void fit(std::vector<std::vector<double>>& data, std::vector<int>& labels) {
        root = buildTree(data, labels, 0);
    }

    std::vector<int> predict(std::vector<std::vector<double>>& testData) {
        std::vector<int> predictions(testData.size());
        #pragma omp parallel for
        for (size_t i = 0; i < testData.size(); i++) {
            predictions[i] = predictSample(root, testData[i]);
        }
        return predictions;
    }

private:
    TreeNode* buildTree(std::vector<std::vector<double>>& data, std::vector<int>& labels, int depth) {
        if (depth >= maxDepth || data.empty()) {
            return createLeaf(labels);
        }

        int bestFeature;
        double bestThreshold;
        std::vector<std::vector<double>> leftData, rightData;
        std::vector<int> leftLabels, rightLabels;

        #pragma omp parallel
        {
            #pragma omp single
            findBestSplit(data, labels, bestFeature, bestThreshold, leftData, leftLabels, rightData, rightLabels);
        }

        TreeNode* node = new TreeNode();
        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;
        node->left = buildTree(leftData, leftLabels, depth + 1);
        node->right = buildTree(rightData, rightLabels, depth + 1);

        return node;
    }

    TreeNode* createLeaf(std::vector<int>& labels) {
        TreeNode* leaf = new TreeNode();
        leaf->label = majorityLabel(labels);
        return leaf;
    }

    int majorityLabel(std::vector<int>& labels) {
        int count0 = std::count(labels.begin(), labels.end(), 0);
        int count1 = labels.size() - count0;
        return (count0 > count1) ? 0 : 1;
    }

    void findBestSplit(std::vector<std::vector<double>>& data, std::vector<int>& labels,
                       int& bestFeature, double& bestThreshold,
                       std::vector<std::vector<double>>& leftData, std::vector<int>& leftLabels,
                       std::vector<std::vector<double>>& rightData, std::vector<int>& rightLabels) {
        double bestGini = std::numeric_limits<double>::max();

        #pragma omp parallel for collapse(2)
        for (size_t feature = 0; feature < data[0].size(); ++feature) {
            for (size_t sample = 0; sample < data.size(); ++sample) {
                double threshold = data[sample][feature];
                std::vector<std::vector<double>> left, right;
                std::vector<int> leftLab, rightLab;

                for (size_t i = 0; i < data.size(); ++i) {
                    if (data[i][feature] <= threshold) {
                        left.push_back(data[i]);
                        leftLab.push_back(labels[i]);
                    } else {
                        right.push_back(data[i]);
                        rightLab.push_back(labels[i]);
                    }
                }

                double gini = calculateGini(leftLab, rightLab);
                #pragma omp critical
                {
                    if (gini < bestGini) {
                        bestGini = gini;
                        bestFeature = feature;
                        bestThreshold = threshold;
                        leftData = left;
                        rightData = right;
                        leftLabels = leftLab;
                        rightLabels = rightLab;
                    }
                }
            }
        }
    }

    double calculateGini(std::vector<int>& left, std::vector<int>& right) {
        double totalSize = left.size() + right.size();
        double leftGini = 1.0, rightGini = 1.0;

        if (!left.empty()) {
            double p0 = (double)std::count(left.begin(), left.end(), 0) / left.size();
            leftGini -= (p0 * p0 + (1 - p0) * (1 - p0));
        }
        if (!right.empty()) {
            double p0 = (double)std::count(right.begin(), right.end(), 0) / right.size();
            rightGini -= (p0 * p0 + (1 - p0) * (1 - p0));
        }

        return (left.size() / totalSize) * leftGini + (right.size() / totalSize) * rightGini;
    }

    int predictSample(TreeNode* node, std::vector<double>& sample) {
        if (!node->left && !node->right) {
            return node->label;
        }
        return (sample[node->featureIndex] <= node->threshold) ? predictSample(node->left, sample) : predictSample(node->right, sample);
    }
};

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    
    DecisionTree dtree(5);
    // TODO change to iris dataset
    std::vector<std::vector<double>> trainData = {{2.5}, {1.5}, {3.0}, {0.5}};
    std::vector<int> trainLabels = {1, 0, 1, 0};
    dtree.fit(trainData, trainLabels);
    
    std::vector<std::vector<double>> testData = {{2.0}, {3.5}};
    std::vector<int> predictions = dtree.predict(testData);
    
    for (int p : predictions) {
        std::cout << "Prediction: " << p << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
    
    return 0;
}
