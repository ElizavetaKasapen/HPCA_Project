#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

using namespace std;

/**
 * This class provides a method to load the Iris dataset from a CSV file.
 */

class DataLoader {
public:
    void load_iris_dataset(const string& filepath, vector<vector<double>>& data, vector<int>& labels);
    void load_wine_dataset(const string& filename, vector<vector<double>>& data, vector<int>& labels);
};

#endif