#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

/**
 * This class loads the Iris dataset from a CSV file
 * Each row of the Iris dataset consists of 4 numerical features followed by a label
 * The function here stores feature values in `data` and class labels in `labels`
 *  
 */

void DataLoader::load_iris_dataset(const string& filepath, vector<vector<double>>& data, vector<int>& labels) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filepath << endl;
        return;
    }

    string line;
    int line_count = 0;
    // Reading the file line by line
    while (getline(file, line)) {
        if (line.empty()) continue; 
        line_count++;

        if (line_count <= 5) {  // Print only first 5 lines
            cout << "Line " << line_count << ": " << line << endl;
        }
        stringstream ss(line);
        string token;
        vector<double> row;

        try {
            // Reading the first 4 columns that will be the numerical features
            for (int i = 0; i < 4; ++i) {
                if (!getline(ss, token, ',')) {
                    throw invalid_argument("Invalid data format");
                }
                row.push_back(stod(token));
            }

            data.push_back(row);

            // Reading the last column that will be the class label
            if (!getline(ss, token, ',')) {
                throw invalid_argument("Invalid label format");
            }

            // The process of class encoding is taking place here by converting string labels to an integer
            if (token == "Iris-setosa") labels.push_back(0);
            else if (token == "Iris-versicolor") labels.push_back(1);
            else if (token == "Iris-virginica") labels.push_back(2);
            else throw invalid_argument("Unknown label: " + token);
        } catch (const invalid_argument& e) {
            cerr << "Error parsing line: " << line << " - " << e.what() << endl;
        }
    }

    file.close();
}

void DataLoader::load_wine_dataset(const string& filename, vector<vector<double>>& data, vector<int>& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }

 else {
     cout << "File opened successfully: " << filename << endl;
}

    string line;
    int line_count = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> features;
        string value;
        line_count++;
        try {
            if (line_count <= 5) {  // Print only first 5 lines
                cout << "Line " << line_count << ": " << line << endl;
            }

            // Read 11 features
            for (int i = 0; i < 11; ++i) {
                getline(ss, value, ';');  // Semicolon-separated values
                features.push_back(stod(value));
            }

            // Read quality score (last column)
            getline(ss, value, ';');
            int quality = stoi(value);

            // Convert quality score into classification labels
            int label;
            if (quality <= 5) label = 0;  // Low quality
            else if (quality <= 7) label = 1;  // Medium quality
            else label = 2;  // High quality

            data.push_back(features);
            labels.push_back(label);
        }
        catch (const invalid_argument& e) {
            cerr << "Error parsing line: " << line << " - " << e.what() << endl;
        }
    }

    file.close();
}