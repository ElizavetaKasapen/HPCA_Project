#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

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
        if (line_count == 0) {
            line_count++; // Skip header line
            continue;
        }
        line_count++;
        try {
        
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