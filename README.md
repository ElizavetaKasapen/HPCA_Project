# DecisionTree_Project

## 📌 Project Overview

This project implements a **Decision Tree** algorithm using three different approaches:

- **CPU (OpenMP-based implementation)**
- **GPU (CUDA-based implementation)**
- **Standard (Sequential implementation)**

The project aims to compare performance differences between these implementations.

---

## 📁 Directory Structure
```
DecisionTree_Project/
│── CPU/                     # OpenMP-based implementation
│   ├── decision_tree.h
│   ├── decision_tree_openmp.cpp
│
│── GPU/                     # CUDA-based implementation
│   ├── decision_tree.h
│   ├── decision_tree_cuda.cu
│
│── standard/                 # Sequential implementation
│   ├── decision_tree.h
│   ├── decision_tree_without_openmp.cpp
│
│── data_loader.cpp           # Data loading functions
│── data_loader.h             # Header for data loading
│── main.cpp                  # Main entry point for execution
│── winequality-white.csv     # Dataset 
│── README.md                 # Project documentation
│── run_experiments.ps1       # Script to run experiments
│── system_info.ps1           # Script to get system info
```
---

## 🛠️ Installation & Compilation

### Prerequisites

- Windows OS
- **C++ Compiler** (GCC, Clang, or MSVC)
- **CUDA Toolkit** (For GPU implementation)
- **OpenMP Support** (For parallel CPU implementation)

### 🔨 Compilation

#### **Standard (Sequential) Version**
```sh
g++ -o main_standard main.cpp data_loader.cpp standard/decision_tree_without_openmp.cpp
```

#### **OpenMP Version (CPU)**
```sh
g++ -o main_openmp -DUSE_OPENMP main.cpp data_loader.cpp CPU/decision_tree_openmp.cpp -fopenmp
```

#### **CUDA Version (GPU)**
```sh
nvcc -o main_cuda -DUSE_CUDA main.cpp data_loader.cpp GPU/decision_tree_cuda.cu
```

---

##  🚀 Usage
Run the compiled executables to compare performance:

#### **Standard Execution**
```sh
 ./main_standard.exe
 ```
#### **OpenMP Execution**
```sh
 ./main_openmp.exe
 ```
#### **CUDA Execution**
```sh
 ./main_cuda.exe
 ```

#### **Run Automated Experiments**
Run the experiments to see and log average execution time and accuracy:

```sh
.\run_experiments.ps1 -logFile <output_logfile_path> -exeFile <execution_file_path>
 ```

- Replace <output_logfile_path> with the desired log file location.
- Replace <execution_file_path> with the path to the executable (e.g., ./main_standard.exe). 
---

## 📊 Dataset: **Wine Quality Prediction**

The dataset used in this project is **`winequality-white.csv`**, which contains **4,898 samples** of white wine samples with physicochemical test results. The goal is to predict wine quality based on its chemical properties.

### **Features (Input Variables)**
1. **Fixed acidity** – Concentration of non-volatile acids.
2. **Volatile acidity** – Concentration of acetic acid (affects taste).
3. **Citric acid** – Found in small amounts, adds freshness.
4. **Residual sugar** – Sugar remaining after fermentation.
5. **Chlorides** – Salt content.
6. **Free sulfur dioxide** – SO₂ in the wine, protects against microbes.
7. **Total sulfur dioxide** – Sum of free and bound SO₂.
8. **Density** – Mass per unit volume, related to alcohol and sugar content.
9. **pH** – Acidity level (low pH = high acidity).
10. **Sulphates** – Antioxidant and antimicrobial agent.
11. **Alcohol** – Percentage of alcohol in the wine.

### **Target Variable (Output)**
12. **Quality** – A score from **0 to 10**, based on sensory evaluations.

### **Classification Labels**
For this project, wine quality is categorized into three classes:
- **Low Quality (0-5) → Label: 0**
- **Medium Quality (6-7) → Label: 1**
- **High Quality (8-10) → Label: 2**

### **Train-Test Split**
The dataset is randomly shuffled and split into **80% training data** and **20% test data**.

---

## 📊 Experiment Results

The following results compare the training time and accuracy of different implementations of the Decision Tree algorithm.

| Implementation       | Training Time (seconds) | Accuracy (%)  |
|----------------------|----------------------:|-------------:|
| **Sequential**       | 28.48988              | 74.32%       |
| **OpenMP (CPU)**     | 4.283573              | 73.69%       |
| **CUDA (GPU)**       | 0.3229036             | 74.37%       |


---
## 📌 Authors
- **Klejda Rrapaj**: k.rrapaj@student.unisi.it
- **Sildi Ricku**: s.ricku@student.unisi.it
- **Yelyzaveta Kasapien**: y.kasapien@student.it