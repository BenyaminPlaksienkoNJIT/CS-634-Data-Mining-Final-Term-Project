#i used Python Version: 3.8.20 and Conda Version: 24.11.3
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd

def calculate_metrics(y_true, y_pred):
    #Confusion matrix to get TP,TN,FP, and FN
    cm = confusion_matrix(y_true, y_pred)
    #print(f'{len(cm)} x {len(cm[0])}') this was just to check shape
    
    #print(cm.shape)
    TP, FN, FP, TN = cm.flatten() if cm.shape == (2,2) else (0,0,0,0)  

    # True Positive Rate
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0  
    # True Negative Rate
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0  
    # False Positive Rate
    FPR = FP / (TN + FP) if (TN + FP) != 0 else 0  
    # False Negative Rate
    FNR = FN / (TP + FN) if (TP + FN) != 0 else 0  
    # Precision
    Precision = TP / (TP + FP) if (TP + FP) != 0 else 0 
    # F1 Measure
    F1 = (2 * Precision * TPR) / (Precision + TPR) if (Precision + TPR) != 0 else 0
    # Accuracy
    Accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) != 0 else 0
    # Balanced Accuracy (BACC)
    Balanced_Accuracy = (TPR + TNR) / 2  
    # Error Rate
    ErrorRate = (FP + FN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) != 0 else 0  
    # True Skill Statistic
    TSS = TPR - FPR  
    # Heidke Skill Score
    HSS = (2 * (TP * TN - FP * FN)) / ((TP + FP) * (FN + TN) + (TP + FN) * (TN + FP)) if (TP + FP) * (FN + TN) + (TP + FN) * (TN + FP) != 0 else 0 
    #Total
    T=TP+FN+TN+FP
    #Total Positive
    P = TP+FN
    #Total Negative
    N =TN+FP
    
    return TP, TN, FP, FN, FPR, FNR, TSS, HSS, Precision, F1, Accuracy, Balanced_Accuracy, ErrorRate, T , P , N

data = pd.read_csv('diabetes.csv')

#Separate features (X) and target variable (y)
X = data.drop(columns=['Outcome'])
y = data['Outcome']  
#Show class distribution/ Data skewing for target outcomes (obviously the data is a bit skewed)
print(y.value_counts(normalize=True))

#Standardize the features (for better model performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Reshape data for LSTM: We need a 3D array (samples, time steps, features)
X_scaled_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

#model setup
rf_model = RandomForestClassifier(n_estimators=100, random_state=420)
gnb_model = GaussianNB()
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model
lstm_model = create_lstm_model((X_scaled_lstm.shape[1], X_scaled_lstm.shape[2]))

#Set up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=420)

#Initialize lists to hold all metrics for each model
metrics_rf_list = []
metrics_gnb_list = []
metrics_lstm_list = []
fold_dfs = []

#Column headers for all models
columns = ["Fold", "Algorithm", "TP", "TN", "FP", "FN", "FPR", "FNR", "TSS", "HSS", 
           "Precision", "F1", "Accuracy", "Balanced_Accuracy", "ErrorRate", "T", "P", "N"]

#Perform KFold cross-validation and calculate metrics for each fold for all models
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled), start=1):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    
    #------------------------------------------------------------------------------------
    
    #Random Forest
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    metrics_rf = calculate_metrics(y_test, y_pred_rf)
    metrics_entry_rf = [fold, "RandomForest", *metrics_rf]
    metrics_rf_list.append(metrics_entry_rf)
    #------------------------------------------------------------------------------------
    
    #Gaussian Naive Bayes
    gnb_model.fit(X_train, y_train)
    y_pred_gnb = gnb_model.predict(X_test)
    metrics_gnb = calculate_metrics(y_test, y_pred_gnb)
    metrics_entry_gnb = [fold, "GaussianNB", *metrics_gnb]
    metrics_gnb_list.append(metrics_entry_gnb)
    #------------------------------------------------------------------------------------
    
    #LSTM 
    X_train_lstm, X_test_lstm = X_scaled_lstm[train_index], X_scaled_lstm[test_index]#Prepare data for LSTM
    lstm_model.fit(X_train_lstm, y_train, epochs=5, batch_size=32, verbose=0)
    y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
    metrics_lstm = calculate_metrics(y_test, y_pred_lstm)
    metrics_entry_lstm = [fold, "LSTM", *metrics_lstm]
    metrics_lstm_list.append(metrics_entry_lstm)
    #------------------------------------------------------------------------------------

    
    fold_df = pd.DataFrame([metrics_entry_rf, metrics_entry_gnb, metrics_entry_lstm], columns=columns)
    fold_dfs.append(fold_df)

#PRINTING tabular format listing all details for easier visualization for each fold and average
for fold, fold_df in enumerate(fold_dfs, start=1):
    print(f"\nMetrics for Fold {fold}:")
    print(fold_df.to_string(index=False))

df_rf = pd.DataFrame(metrics_rf_list, columns=columns)
df_gnb = pd.DataFrame(metrics_gnb_list, columns=columns)
df_lstm = pd.DataFrame(metrics_lstm_list, columns=columns)

print("\nMetrics Across All Folds for Random Forest:")
print(df_rf.to_string(index=False))
print("\nMetrics Across All Folds for Gaussian Naive Bayes:")
print(df_gnb.to_string(index=False))
print("\nMetrics Across All Folds for LSTM:")
print(df_lstm.to_string(index=False))

df_rf_no_fold = df_rf.drop(columns=['Fold'])
df_gnb_no_fold = df_gnb.drop(columns=['Fold'])
df_lstm_no_fold = df_lstm.drop(columns=['Fold'])
average_metrics_rf = df_rf_no_fold.mean(numeric_only=True)
average_metrics_gnb = df_gnb_no_fold.mean(numeric_only=True)
average_metrics_lstm = df_lstm_no_fold.mean(numeric_only=True)

print("\nAverage Metrics Across All Folds for Random Forest:")
print(average_metrics_rf)
print("\nAverage Metrics Across All Folds for Gaussian Naive Bayes:")
print(average_metrics_gnb)
print("\nAverage Metrics Across All Folds for LSTM:")
print(average_metrics_lstm)

