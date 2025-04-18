# Diabetes Classification Models with Metric Evaluation

## Data Source
The diabetes prediction dataset used in this project is sourced from:  
[Kaggle: Diabetes Data Set](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)

## Overview
This program compares the performance of three classification algorithms on a medical dataset (diabetes prediction), using custom metric evaluation across 10-fold cross-validation:

- **Random Forest Classifier**: An ensemble method that builds multiple decision trees and combines their outputs.
  
- **Gaussian Naive Bayes**: A probabilistic classifier based on Bayes' Theorem with the assumption of feature independence.
  
- **LSTM Neural Network**: A recurrent neural network (RNN) architecture suitable for sequence prediction, adapted here for tabular classification with one time step.

The evaluation includes a range of metrics beyond standard accuracy, such as True Skill Statistic (TSS) and Heidke Skill Score (HSS).

## Prerequisites
Ensure you have the following software and libraries installed:

### Required Versions
- Python Version: 3.8.20
- Conda Version: 24.11.3

### Required Libraries
Install the necessary Python libraries using pip:

```sh
pip install numpy pandas scikit-learn tensorflow
```
## How It Works
The program follows these steps:

### Data Loading & Preprocessing:
- Reads the `diabetes.csv` dataset.
- Separates features (X) and labels (y).
- Standardizes features using `StandardScaler`.
- Reshapes the data for LSTM input format (3D tensor).

### Model Initialization:
- Prepares:
  - Random Forest
  - Gaussian Naive Bayes
  - LSTM network using Keras

### Cross-Validation Setup:
- Uses 10-fold cross-validation (`KFold` from `sklearn.model_selection`)
- For each fold:
  - Trains all three models
  - Evaluates all three models

### Metric Evaluation:
Calculates confusion matrix and derives:
- Basic counts:
  - TP, TN, FP, FN
- Rates and scores:
  - FPR, FNR
  - TSS, HSS
  - Precision, F1 Score
  - Accuracy, Balanced Accuracy
  - Error Rate

These metrics are calculated via a custom function for each fold and model.

### Results Display:
- Prints fold-wise metrics in tabular format
- Aggregates results and shows average metrics across all folds for each model

## Running the Program

### Running on Command Line
Execute the script using:

```sh
python Benyamin_Plaksienko_Final_Project.py
```
This will:
- Run the complete process
- Display metrics for each fold  
- Show the overall average metrics

### Running on Jupyter Notebook
The code can be adapted for interactive analysis:

```sh
jupyter notebook
```
## Output

The program displays:

### Per-Fold Metrics
- Formatted table showing metrics for each algorithm across all folds

### Confusion Matrix Statistics
- **Accuracy**
- **Precision**  
- **Recall**
- **F1 Score**
- **True Skill Statistic (TSS)**
- **Heidke Skill Score (HSS)**
- **Balanced Accuracy**  
- **Error Rate**

### Average Performance Metrics
Calculated across all folds for:
- **Random Forest**
- **Gaussian Naive Bayes**  
- **LSTM** neural network

---

**Author**: Benyamin Plaksienko