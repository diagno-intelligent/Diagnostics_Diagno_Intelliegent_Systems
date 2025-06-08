def ens():
    import os
    import numpy as np
    import pandas as pd
    import pickle
    import joblib
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    # Optional, only for the commented-out confusion matrix section
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import confusion_matrix


    # Load data
    file_path = "./input_feature_unlabeled.csv"
    #file_path = "D:/clavicle_new_mes_reg_score3/Female_3pt_1pt_new_measurement _whole.xlsx"
    data1=data  = pd.read_csv(file_path)
    #data  = pd.read_excel(file_path, engine='openpyxl') #, sheet_name='train'
    #data = pd.read_csv("D:/Bone_strength_Measurement(16.11.2022)/Deep_feature_code_results/2nd_phase/feature_model_2nd_phase_eff_B3_20d_ADAM_sq_aug_score3_BOTH_FX1_16b_f1_new_D7.csv")#feature2.csv")#feature_resample2_T_full.csv
    X = data.iloc[:1,1:]   #independent columns
    #print(X)
    #y = data.iloc[:1,1]    #target column i.e w_m_cl_b WITHOUT_TRIM cl_b M_cl_b
    ##print(y)

    X1=X
    #y1=y

    list_id = data1.iloc[:1, 0]
    list_ID = data1.iloc[:1, 0]
    #print("\nrf_mi")
    def scale_datasets(X, scaler_path):
        ##print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)

    # Define the path to the scaler file
    scaler_path = r'./selected_models/3a_scaler_ALL_FEATURE_5m_SCORE_rf_mutual_info_classif_BOTH__min_max_w_fec.pkl'

    # Print the path to verify
    ##print(f"Scaler path: {repr(scaler_path)}")
    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))

    #X_test1 = scale_datasets(X1)

    from sklearn.linear_model import Lasso

        #X_train_selected = rfe.transform(X_train)
    with open(f'./selected_models/3a_selected_features_5m_rf_mutual_info_classif_BOTH_w_fec_101.txt', 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))

    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

    selected_feature_names = X1.columns[selected_feature_indices]
    ##print("Selected features:", selected_feature_names)

    loaded_SVM_model = joblib.load(f"./selected_models/3a_5m_BOTH_rf_model_mutual_info_classif_fec_101_train_acc1.0_test_acc0.9652103559870551.pkl")


    y_pred1=rf_mi_5m = loaded_SVM_model.predict(X_test_selected1)
    #print('y_pred1',y_pred1)
    unique_labels = np.unique(y_pred1)
    #print("Unique Labels:", unique_labels)
    for i in range (0,len(y_pred1)):
        #print('y_pred1[i]',y_pred1[i])
        if y_pred1[i]==1:
            y_pred1[i]=1
        if y_pred1[i]==2:
            y_pred1[i]=2
        if y_pred1[i]==3:
            y_pred1[i]=3
        if y_pred1[i]==4:
            y_pred1[i]=4
        if y_pred1[i]==0:
            y_pred1[i]=0
    rf_mi_5m=y_pred1
    #############33
    #print("\nxgb_chi2")
    def scale_datasets(X, scaler_path):
        ##print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)

    # Define the path to the scaler file
    scaler_path = r'./selected_models/1a_scaler_ALL_FEATURE_5m_xgb_chi2__min_max_K_{k}.pkl'

    # Print the path to verify
    ##print(f"Scaler path: {repr(scaler_path)}")
    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))

    #X_test1 = scale_datasets(X1)

    from sklearn.linear_model import Lasso

        #X_train_selected = rfe.transform(X_train)
    with open(f'./selected_models/1a_selected_features_5m_XG_chi2_k100.txt', 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))

    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

    selected_feature_names = X1.columns[selected_feature_indices]
    ##print("Selected features:", selected_feature_names)

    loaded_SVM_model = joblib.load(f"./selected_models/1a_5m_XG_chi2_fec_100_train_acc1.0_test_acc0.9644012944983819.pkl")


    y_pred1=xgb_chi2_5m = loaded_SVM_model.predict(X_test_selected1)
    #print(y_pred1)
    for i in range (0,len(y_pred1)):
        if y_pred1[i]==1:
            y_pred1[i]=1
        if y_pred1[i]==2:
            y_pred1[i]=2
        if y_pred1[i]==3:
            y_pred1[i]=3
        if y_pred1[i]==4:
            y_pred1[i]=4
        if y_pred1[i]==0:
            y_pred1[i]=0
    xgb_chi2_5m=y_pred1

    #################

    #print("\n xgb_mi")
    def scale_datasets(X, scaler_path):
        ##print(f"Attempting to open scaler file at: {repr(scaler_path)}")
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return scaler.transform(X)

    # Define the path to the scaler file
    scaler_path = r'./selected_models/2a_scaler_ALL_FEATURE_5m_xgb_mutual_info_classif__min_max_K_{k}.pkl'

    # Print the path to verify
    ##print(f"Scaler path: {repr(scaler_path)}")
    X_test1 = pd.DataFrame(scale_datasets(X1, scaler_path))

    #X_test1 = scale_datasets(X1)

    from sklearn.linear_model import Lasso

        #X_train_selected = rfe.transform(X_train)
    with open(f'./selected_models/2a_selected_features_5m_XG_mutual_info_classif_k298.txt', 'r') as file:
        selected_feature_indices = list(map(int, file.read().split(',')))

    X_test_selected1 = X_test1.iloc[:, selected_feature_indices]

    selected_feature_names = X1.columns[selected_feature_indices]
    ##print("Selected features:", selected_feature_names)

    loaded_SVM_model = joblib.load(f"./selected_models/2a_5m_XG_mutual_info_classif_fec_298_train_acc1.0_test_acc0.9644012944983819.pkl")


    y_pred1=xgb_mi_5m = loaded_SVM_model.predict(X_test_selected1)
    #print(y_pred1)
    for i in range (0,len(y_pred1)):
        if y_pred1[i]==1:
            y_pred1[i]=1
        if y_pred1[i]==2:
            y_pred1[i]=2
        if y_pred1[i]==3:
            y_pred1[i]=3
        if y_pred1[i]==4:
            y_pred1[i]=4
        if y_pred1[i]==0:
            y_pred1[i]=0
    xgb_mi_5m=y_pred1

    ############### ens_STACK
    #print('/n')
    #print('stacked_ML_5m_nr')
    preds_model1=rf_mi_5m
    preds_model2=xgb_chi2_5m
    preds_model3=xgb_mi_5m

    # Combine predictions into a feature matrix
    X_stack = np.column_stack((preds_model1, preds_model2, preds_model3))
    ##print("Predicted Value:", predicted_value)
    loaded_model = joblib.load('stacked_ensemble_model_ML_5m_cl_F.pkl')
    predicted_value1 = loaded_model.predict(X_stack)
    #print('predicted_value1',predicted_value1)
    probabilities = loaded_model.predict_proba(X_stack)[0]
    #print('probabilities',probabilities)
    #### for single prediction
    predicted_value1 = int(loaded_model.predict(X_stack)[0])
    #print('predicted_value1',predicted_value1)

    if predicted_value1==0:
        Final_prediction="COPD"
        probabilities = probabilities[0]
    elif predicted_value1==1:
        Final_prediction="Lung Cancer"
        probabilities = probabilities[1]
    elif predicted_value1==2:
        Final_prediction="Normal"
        probabilities = probabilities[2]
    elif predicted_value1==3:
        Final_prediction="TB-Positive"
        probabilities=probabilities[3]
    elif predicted_value1==4:
        Final_prediction="Silicosis"
        probabilities=probabilities[4]
    print('Final_prediction:',Final_prediction)
    return(Final_prediction,predicted_value1,probabilities)
##############  ST_ENS
##cm_test1 = confusion_matrix(y1, predicted_value1)
##cm_test_df1 = pd.DataFrame(cm_test1, index=['copd','lc','nr','tb','si'], columns=['copd','lc','nr','tb','si'])
##plt.figure(figsize=(5, 4))
##sns.heatmap(cm_test_df1, annot=True)
##plt.title(f"ST_ENS whole_5m_Confusion Matrix_5 class(score-5m)_Test")
##plt.ylabel('Actual Values')
##plt.xlabel('Predicted Values')
##
##class_wise_accuracy1 = np.diag(cm_test1) / cm_test1.sum(axis=1)
###print("\nST_ENS Whole data Class-wise accuracy:")
##total_acc = 0
##for i, acc in enumerate(class_wise_accuracy1):
##    #print(f"Class {i}: {acc*100:.2f}%")
##    total_acc += acc
##
##total_accuracy1 = total_acc / 5
###print(f"ST_ENS Whole case accuracy: {total_accuracy1*100:.2f}%")




