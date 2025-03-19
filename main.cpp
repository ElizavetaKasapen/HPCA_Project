#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <numeric>
#include "data_loader.h"

#ifdef USE_CUDA
#include "GPU/decision_tree.h"
#define IMPLEMENTATION "GPU"
#elif defined USE_OPENMP
#include "CPU/decision_tree.h"
#define IMPLEMENTATION "CPU"
#else
#include "standard/decision_tree.h"
#define IMPLEMENTATION "standard"
#endif

using namespace std;

void split_dataset(const vector<vector<double>> &data, const vector<int> &labels,
                   vector<vector<double>> &train_data, vector<int> &train_labels,
                   vector<vector<double>> &test_data, vector<int> &test_labels,
                   double train_ratio = 0.8)
{
    random_device rd;
    mt19937 g(rd());
    vector<size_t> indices(data.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), g);
    size_t train_size = static_cast<size_t>(data.size() * train_ratio);

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (i < train_size)
        {
            train_data.push_back(data[indices[i]]);
            train_labels.push_back(labels[indices[i]]);
        }
        else
        {
            test_data.push_back(data[indices[i]]);
            test_labels.push_back(labels[indices[i]]);
        }
    }
}

double calculate_accuracy(const vector<int> &true_labels, const vector<int> &predicted_labels)
{
    if (true_labels.size() != predicted_labels.size())
    {
        throw invalid_argument("True labels and predicted labels must have the same size.");
    }
    size_t correct = 0;
    for (size_t i = 0; i < true_labels.size(); ++i)
    {
        if (true_labels[i] == predicted_labels[i])
        {
            correct++;
        }
    }
    return static_cast<double>(correct) / true_labels.size();
}

int main()
{
    DataLoader loader;
    vector<vector<double>> data;
    vector<int> labels;

    cout << "Loading dataset..." << endl;
    loader.load_wine_dataset("winequality-white.csv", data, labels);
    cout << "Loaded " << data.size() << " samples." << endl;

    vector<vector<double>> train_data, test_data;
    vector<int> train_labels, test_labels;

    double total_time = 0.0;
    double total_accuracy = 0.0;

    split_dataset(data, labels, train_data, train_labels, test_data, test_labels);

    DecisionTree tree;

    // Train part
    auto start = chrono::high_resolution_clock::now();
    tree.fit(train_data, train_labels);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> training_time = end - start;
    cout << "Training time: " << training_time.count() << " seconds\n";

    // Predict part
    vector<int> predicted_labels;
    for (const auto &sample : test_data)
    {
        predicted_labels.push_back(tree.predict(sample));
    }

    double accuracy = calculate_accuracy(test_labels, predicted_labels);
    cout << "Accuracy: " << accuracy * 100 << "%\n";

    cout << "Experiment completed." << endl;
    return 0;
}
