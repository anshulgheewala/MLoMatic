# import sys
# import json
# import pickle
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, precision_score, recall_score, f1_score
# from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
# from sklearn.svm import SVC, SVR
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from xgboost import XGBClassifier
# from sklearn.decomposition import PCA
# from imblearn.over_sampling import SMOTE
# import warnings

# warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
# warnings.filterwarnings("ignore", category=FutureWarning)

# def main():
#     file_path = sys.argv[1]
#     target_column = sys.argv[2]
#     problem_type = sys.argv[3].lower()
#     selected_models_str = sys.argv[4]
#     selected_models = json.loads(selected_models_str)
#     replace_column = sys.argv[5]
#     find_value = sys.argv[6]
#     replace_value = sys.argv[7]

#     print(find_value, file=sys.stderr)
#     print(replace_value, file=sys.stderr)

#     if replace_value and isinstance(replace_value, str) and replace_value.lower() == 'null':
#         replace_value = None

#     df = pd.read_csv(file_path)

#     if replace_column and replace_column in df.columns:
#         print(df[replace_column].head(), file=sys.stderr)
#         df[replace_column] = df[replace_column].replace(find_value, replace_value)
#         print(df[replace_column].head(), file=sys.stderr)
#     else:
#         print(f"[Info] replace_column '{replace_column}' not provided or not found in dataset.", file=sys.stderr)

#     # if replace_value.lower() == 'null':
#     #     replace_value = None

#     # df = pd.read_csv(file_path)
#     # print(df[replace_column].head(), file=sys.stderr)

#     # if replace_column and replace_column in df.columns:
#     #     df[replace_column] = df[replace_column].replace(find_value, replace_value)

#     #     # before_values = df[replace_column].copy()

#     #     df[replace_column] = df[replace_column].replace(find_value, replace_value)
#     #     print(df[replace_column].head(), file=sys.stderr)

#     # Handle missing values
#     for col in df.columns:
#         if df[col].dtype in ['float64', 'int64']:
#             df[col] = df[col].fillna(df[col].mean())
#         else:
#             df[col] = df[col].fillna(df[col].mode()[0])

#     # Remove commas in numeric strings
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             try:
#                 df[col] = df[col].str.replace(',', '').astype(float)
#             except:
#                 pass

#     X = df.drop(columns=[target_column])
#     y = df[target_column]
#     X = pd.get_dummies(X, drop_first=True)
#     # the next line is a chanege
#     feature_columns = X.columns.tolist() 
#     # Encode target for classification
#     if problem_type == 'classification' and y.dtype == 'object':
#         le = LabelEncoder()
#         y = le.fit_transform(y)

#     # -------------------------
#     # Train/Test Split
#     # -------------------------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=42
#     )

#     # -------------------------
#     # SMOTE only on training set
#     # -------------------------
#     # if problem_type == 'classification':
#     #     smote = SMOTE(random_state=42)
#     #     X_train, y_train = smote.fit_resample(X_train, y_train)

#     if problem_type == "classification":
    
#         # Count class distribution
#         class_counts = pd.Series(y_train).value_counts().sort_values(ascending=False)
#         max_count = class_counts.iloc[0]
#         mean_count = int(class_counts.mean())

#         imbalance_factor = max_count / (mean_count + 1e-9)
#         imbalance_threshold = 1.5  # if max/mean > this, we consider imbalanced
#         balance_mode = 'to_moderate'  # 'to_max' for full balance

#         if imbalance_factor > imbalance_threshold:
#             if balance_mode == 'to_max':
#                 target_count = int(max_count)
#             else:  # to_moderate
#                 target_count = min(int(mean_count * 1.5), int(max_count))

#             # Only balance classes below target_count
#             sampling_strategy = {int(cls): target_count for cls, cnt in class_counts.items() if cnt < target_count}

#             if sampling_strategy:
#                 smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
#                 X_train, y_train = smote.fit_resample(X_train, y_train)
#                 print(f"[SMOTE Applied] Before: {class_counts.to_dict()}", file=sys.stderr)
#                 print(f"[SMOTE Applied] After : {pd.Series(y_train).value_counts().to_dict()}", file=sys.stderr)
#             else:
#                 print("[SMOTE Skipped] No minority class below target count.", file=sys.stderr)
#     else:
#         print(f"[SMOTE Skipped] imbalance_factor={imbalance_factor:.2f} <= threshold {imbalance_threshold}", file=sys.stderr)

#     # -------------------------
#     # Scaling + PCA without leakage
#     # -------------------------
#     numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
#     continuous_cols = [
#         col for col in numeric_cols
#         if X_train[col].nunique() > 5
#     ]

#     scaler = None
#     pca = None
#     pca_cols = continuous_cols

#     if continuous_cols:
#         scaler = StandardScaler()
#         X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
#         X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

#         pca = PCA(n_components=0.95)
#         X_train_pca = pca.fit_transform(X_train[continuous_cols])
#         X_test_pca = pca.transform(X_test[continuous_cols])

#         pca_col_names = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]

#         X_train = X_train.drop(columns=continuous_cols).reset_index(drop=True)
#         X_test = X_test.drop(columns=continuous_cols).reset_index(drop=True)

#         X_train = pd.concat([X_train, pd.DataFrame(X_train_pca, columns=pca_col_names)], axis=1)
#         X_test = pd.concat([X_test, pd.DataFrame(X_test_pca, columns=pca_col_names)], axis=1)

#         print("\n[Scaling + PCA Applied]", file=sys.stderr)
#         print("Scaled columns:", list(continuous_cols), file=sys.stderr)
#         print("PCA components kept:", pca.n_components_, file=sys.stderr)
#         print("Explained variance ratio:", pca.explained_variance_ratio_, file=sys.stderr)
#         print("----------------------\n", file=sys.stderr)

#     # -------------------------
#     # Models
#     # -------------------------
#     classification_models = {
#         'Logistic Regression': LogisticRegression(max_iter=1000),
#         'Random Forest': RandomForestClassifier(random_state=42),
#         'Support Vector Machine': SVC(probability=True, random_state=42),
#         'AdaBoost': AdaBoostClassifier(random_state=42),
#         'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
#         'Decission Tree': DecisionTreeClassifier(),
#     }

#     regression_models = {
#         'Linear Regression': LinearRegression(),
#         'Ridge': Ridge(),
#         'Lasso': Lasso(),
#         'Random Forest': RandomForestRegressor(random_state=42),
#         'Support Vector Machine': SVR(),
#         'Decission Tree': DecisionTreeRegressor()
#     }

#     classification_params = {
#         'Logistic Regression': {
#             'C': [0.01, 0.1, 1, 10],
#             'solver': ['lbfgs', 'liblinear']
#         },
#         'Random Forest': {
#             'n_estimators': [100, 200],
#             'max_depth': [5, 10],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         },
#         'Support Vector Machine': {
#             'C': [0.1, 1, 10],
#             'kernel': ['linear', 'rbf'],
#             'gamma': ['scale', 'auto']
#         },
#         'AdaBoost': {
#             'n_estimators': [50, 100, 200],
#             'learning_rate': [0.01, 0.1, 1]
#         },
#         'XGBoost': {
#             'n_estimators': [100, 200],
#             'max_depth': [3, 5, 7],
#             'learning_rate': [0.01, 0.1, 0.3],
#             'subsample': [0.8, 1.0]
#         },
#         'Decision Tree': {
#             'criterion': ['gini', 'entropy'],
#             'splitter': ['best', 'random'],
#             'max_depth': [None, 10, 20, 30, 40, 50],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4],
#             'max_features': [None, 'sqrt', 'log2']
#         }
#     }

#     regression_params = {
#         'Linear Regression': {},
#         'Ridge': {
#             'alpha': [0.1, 1.0, 10.0]
#         },
#         'Lasso': {
#             'alpha': [0.1, 1.0, 10.0]
#         },
#         'Random Forest': {
#             'n_estimators': [100, 200],
#             'max_depth': [5, 10],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         },
#         'Support Vector Machine': {
#             'C': [0.1, 1, 10],
#             'kernel': ['linear', 'rbf'],
#             'gamma': ['scale', 'auto']
#         },
#         'Decision Tree': {
#             'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
#             'splitter': ['best', 'random'],
#             'max_depth': [None, 10, 20, 30, 40, 50],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4],
#             'max_features': [None, 'sqrt', 'log2']
#         }
#     }

#     if problem_type == 'classification':
#         model_map = classification_models
#         param_grids = classification_params
#         scoring_metric = 'accuracy'
#     else:
#         model_map = regression_models
#         param_grids = regression_params
#         scoring_metric = 'r2'

#     if "best_one" in selected_models:
#         selected_models = list(model_map.keys())

#     results = {}
#     best_model_info = {'model_name': None, 'score': -1, 'model_object': None, 'best_params': None}

#     for model_name in selected_models:
#         if model_name not in model_map:
#             continue

#         model = model_map[model_name]
#         params = param_grids.get(model_name, {})

#         if params:
#             grid = GridSearchCV(
#                 estimator=model,
#                 param_grid=params,
#                 cv=5,
#                 scoring=scoring_metric,
#                 n_jobs=3
#             )
#             grid.fit(X_train, y_train)
#             best_estimator = grid.best_estimator_
#             best_params = grid.best_params_
#             cv_score = grid.best_score_
#         else:
#             best_estimator = model
#             best_estimator.fit(X_train, y_train)
#             best_params = {}
#             cv_score = cross_val_score(best_estimator, X_train, y_train, cv=5, scoring=scoring_metric).mean()

#         train_score = best_estimator.score(X_train, y_train)
#         test_score = best_estimator.score(X_test, y_test)

#         generalization_gap = abs(train_score - test_score)
#         if generalization_gap > 0.1:
#             test_score -= generalization_gap * 0.5

#         if problem_type == 'classification':
#             y_pred = best_estimator.predict(X_test)
#             cm = confusion_matrix(y_test, y_pred).tolist()
#             precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#             recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
#             f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
#         else:
#             cm = None
#             precision = None
#             recall = None
#             f1 = None

#         results[model_name] = {
#             'cv_score': cv_score,
#             'train_score': train_score,
#             'test_score': test_score,
#             'generalization_gap': generalization_gap,
#             'best_params': best_params,
#             'confusion_matrix': cm,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1
#         }

#         if test_score > best_model_info['score']:
#             best_model_info = {
#                 'model_name': model_name,
#                 'score': test_score,
#                 'model_object': best_estimator,
#                 'best_params': best_params
#             }

#     model_filename = f'model_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
#     model_path = f'uploads/{model_filename}'
#     # with open(model_path, 'wb') as f:
#     #     pickle.dump({
#     #         'model': best_model_info['model_object'],
#     #         'scaler': scaler,
#     #         'pca': pca,
#     #         'pca_cols': pca_cols
#     #     }, f)

#     with open(model_path, 'wb') as f:
#         pickle.dump({
#             'model': best_model_info['model_object'],
#             'scaler': scaler,
#             'pca': pca,
#             'pca_cols': pca_cols,  # columns PCA was applied to
#             'feature_columns': feature_columns
#         }, f)
#         # pickle.dump({'model':best_model_info['model_object'], 'scaler': scaler}, f)

#     output = {
#         'best_model': best_model_info['model_name'],
#         'metric_name': 'Accuracy' if problem_type == 'classification' else 'R-squared',
#         'score': best_model_info['score'],
#         'best_params': best_model_info['best_params'],
#         'model_path': model_path,
#         'all_results': results
#     }

#     print(json.dumps(output))

# if __name__ == '__main__':
#     main()



# # import sys
# # import json
# # import pickle
# # import pandas as pd
# # from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, precision_score, recall_score, f1_score
# # from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
# # from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# # from sklearn.svm import SVC, SVR
# # from sklearn.ensemble import AdaBoostClassifier
# # from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
# # from xgboost import XGBClassifier
# # from sklearn.decomposition import PCA


# # import warnings

# # warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
# # warnings.filterwarnings("ignore", category=FutureWarning)

# # def main():
# #     file_path = sys.argv[1]
# #     target_column = sys.argv[2]
# #     problem_type = sys.argv[3].lower()
# #     selected_models_str = sys.argv[4]
# #     selected_models = json.loads(selected_models_str)
# #     replace_column = sys.argv[5]
# #     find_value = sys.argv[6]
# #     replace_value = sys.argv[7]

# #     print(find_value,file=sys.stderr)
# #     print(replace_value,file=sys.stderr)


# #     if replace_value.lower() == 'null':
# #         replace_value = None

# #     df = pd.read_csv(file_path)
# #     print(df[replace_column].head(), file=sys.stderr)

# #     if replace_column and replace_column in df.columns:
# #         df[replace_column] = df[replace_column].replace(find_value, replace_value)
    

# #     before_values = df[replace_column].copy()

# # # Replace
# #     df[replace_column] = df[replace_column].replace(find_value, replace_value)

# #     print(df[replace_column].head(), file=sys.stderr)


# #     # Handle missing values
# #     for col in df.columns:
# #         if df[col].dtype in ['float64', 'int64']:
# #             df[col] = df[col].fillna(df[col].mean())
# #         else:
# #             df[col] = df[col].fillna(df[col].mode()[0])

# #     # Remove commas in numeric strings
# #     for col in df.columns:
# #         if df[col].dtype == 'object':
# #             try:
# #                 df[col] = df[col].str.replace(',', '').astype(float)
# #             except:
# #                 pass

# #     X = df.drop(columns=[target_column])
# #     y = df[target_column]
# #     X = pd.get_dummies(X, drop_first=True)

# #     # print(df[replace_column],file=sys.stderr)


# #     # Encode target for classification
# #     if problem_type == 'classification' and y.dtype == 'object':
# #         le = LabelEncoder()
# #         y = le.fit_transform(y)

# #     # changes started here
# #     numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
# #     continuous_cols = [
# #         col for col in numeric_cols
# #         if X[col].nunique() > 5
# #     ]
# #     # print("Mean before scaling:", X[continuous_cols].mean().round(4).to_list(), file=sys.stderr)
# #     # print("Std before scaling:", X[continuous_cols].std().round(4).to_list(), file=sys.stderr)

# #     scaler = None
# #     pca = None
# #     pca_cols = continuous_cols 

# #     if continuous_cols:
# #         scaler = StandardScaler()
# #         X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# #         pca = PCA(n_components=0.95)  # Keep 95% variance
# #         X_pca = pca.fit_transform(X[continuous_cols])
# #         pca_col_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
# #         X = X.drop(columns=continuous_cols).reset_index(drop=True)
# #         X = pd.concat([X, pd.DataFrame(X_pca, columns=pca_col_names)], axis=1)
# #         print("\n[Scaling + PCA Applied]", file=sys.stderr)
# #         print("Scaled columns:", list(continuous_cols), file=sys.stderr)
# #         print("PCA components kept:", pca.n_components_, file=sys.stderr)
# #         print("Explained variance ratio:", pca.explained_variance_ratio_, file=sys.stderr)
# #         print("----------------------\n", file=sys.stderr)
# #     #     print("\n[Scaling Applied]", file=sys.stderr)
# #     #     print("Scaled columns:", list(continuous_cols), file=sys.stderr)
# #     #     print("Mean after scaling:", X[continuous_cols].mean().round(4).to_list(), file=sys.stderr)
# #     #     print("Std after scaling:", X[continuous_cols].std().round(4).to_list(), file=sys.stderr)
# #     # #     print("----------------------\n")
# #     # else:
# #     #     print("No Scaling has been done",file=sys.stderr)

# #     # changes ended here

# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.25, random_state=42
# #     )

# #     # Model dictionaries
# #     classification_models = {
# #         'Logistic Regression': LogisticRegression(max_iter=1000),
# #         'Random Forest': RandomForestClassifier(random_state=42),
# #         'Support Vector Machine': SVC(probability=True, random_state=42),
# #         'AdaBoost': AdaBoostClassifier(random_state=42),
# #         'XGBoost': XGBClassifier(eval_metric='logloss',  random_state=42),
# #         'Decission Tree': DecisionTreeClassifier(),
# #     }

# #     regression_models = {
# #         'Linear Regression': LinearRegression(),
# #         'Ridge': Ridge(),
# #         'Lasso': Lasso(),
# #         'Random Forest': RandomForestRegressor(random_state=42),
# #         'Support Vector Machine': SVR(),
# #         'Decission Tree': DecisionTreeRegressor()
# #     }

# #     # Parameter grids with limits to avoid overfitting
# #     classification_params = {
# #         'Logistic Regression': {
# #             'C': [0.01, 0.1, 1, 10],
# #             'solver': ['lbfgs', 'liblinear']
# #         },
# #         'Random Forest': {
# #             'n_estimators': [100, 200],
# #             'max_depth': [5, 10],
# #             'min_samples_split': [2, 5],
# #             'min_samples_leaf': [1, 2]
# #         },
# #         'Support Vector Machine': {
# #             'C': [0.1, 1, 10],
# #             'kernel': ['linear', 'rbf'],
# #             'gamma': ['scale', 'auto']
# #         }
# #         ,
# #     'AdaBoost': {
# #         'n_estimators': [50, 100, 200],
# #         'learning_rate': [0.01, 0.1, 1]
# #     },
# #     'XGBoost': {
# #         'n_estimators': [100, 200],
# #         'max_depth': [3, 5, 7],
# #         'learning_rate': [0.01, 0.1, 0.3],
# #         'subsample': [0.8, 1.0]
# #     },
# #     'Decision Tree': {
# #         'criterion': ['gini', 'entropy'],
# #         'splitter': ['best', 'random'],
# #         'max_depth': [None, 10, 20, 30, 40, 50],
# #         'min_samples_split': [2, 5, 10],
# #         'min_samples_leaf': [1, 2, 4],
# #         'max_features': [None, 'sqrt', 'log2']
# #     }
# #     }

# #     regression_params = {
# #         'Linear Regression': {},
# #         'Ridge': {
# #             'alpha': [0.1, 1.0, 10.0]
# #         },
# #         'Lasso': {
# #             'alpha': [0.1, 1.0, 10.0]
# #         },
# #         'Random Forest': {
# #             'n_estimators': [100, 200],
# #             'max_depth': [5, 10],
# #             'min_samples_split': [2, 5],
# #             'min_samples_leaf': [1, 2]
# #         },
# #         'Support Vector Machine': {
# #             'C': [0.1, 1, 10],
# #             'kernel': ['linear', 'rbf'],
# #             'gamma': ['scale', 'auto']
# #         },
# #         'Decision Tree': {
# #             'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
# #             'splitter': ['best', 'random'],
# #             'max_depth': [None, 10, 20, 30, 40, 50],
# #             'min_samples_split': [2, 5, 10],
# #             'min_samples_leaf': [1, 2, 4],
# #             'max_features': [None, 'sqrt', 'log2']
# #         }
# #     }

# #     # Choose correct model set
# #     if problem_type == 'classification':
# #         model_map = classification_models
# #         param_grids = classification_params
# #         scoring_metric = 'accuracy'
# #     else:
# #         model_map = regression_models
# #         param_grids = regression_params
# #         scoring_metric = 'r2'

# #     # If best_one selected, run all models of correct type
# #     if "best_one" in selected_models:
# #         selected_models = list(model_map.keys())

# #     results = {}
# #     best_model_info = {'model_name': None, 'score': -1, 'model_object': None, 'best_params': None}

# #     # Train models
# #     for model_name in selected_models:
# #         if model_name not in model_map:
# #             continue

# #         model = model_map[model_name]
# #         params = param_grids.get(model_name, {})

# #         # Hyperparameter tuning with CV
# #         if params:
# #             grid = GridSearchCV(
# #                 estimator=model,
# #                 param_grid=params,
# #                 cv=5,
# #                 scoring=scoring_metric,
# #                 n_jobs=3
# #             )
# #             grid.fit(X_train, y_train)
# #             best_estimator = grid.best_estimator_
# #             best_params = grid.best_params_
# #             cv_score = grid.best_score_
# #         else:
# #             best_estimator = model
# #             best_estimator.fit(X_train, y_train)
# #             best_params = {}
# #             cv_score = cross_val_score(best_estimator, X_train, y_train, cv=5, scoring=scoring_metric).mean()

# #         # Evaluate
# #         train_score = best_estimator.score(X_train, y_train)
# #         test_score = best_estimator.score(X_test, y_test)

# #         # Overfitting check
# #         generalization_gap = abs(train_score - test_score)
# #         if generalization_gap > 0.1:
# #             test_score -= generalization_gap * 0.5  # penalize overfit models
        
# #         # Additional metrics for classification
# #         if problem_type == 'classification':
# #             y_pred = best_estimator.predict(X_test)
# #             cm = confusion_matrix(y_test, y_pred).tolist()  # convert to list for JSON
# #             precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
# #             recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
# #             f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
# #         else:
# #             cm = None
# #             precision = None
# #             recall = None
# #             f1 = None

# #         results[model_name] = {
# #             'cv_score': cv_score,
# #             'train_score': train_score,
# #             'test_score': test_score,
# #             'generalization_gap': generalization_gap,
# #             'best_params': best_params,
# #             'confusion_matrix': cm,
# #             'precision': precision,
# #             'recall': recall,
# #             'f1_score': f1
# #         }

# #         if test_score > best_model_info['score']:
# #             best_model_info = {
# #                 'model_name': model_name,
# #                 'score': test_score,
# #                 'model_object': best_estimator,
# #                 'best_params': best_params
# #             }

# #     # Save best model
# #     model_filename = f'model_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
# #     model_path = f'uploads/{model_filename}'
# #     with open(model_path, 'wb') as f:
# #         pickle.dump({
# #             'model': best_model_info['model_object'],
# #             'scaler': scaler,
# #             'pca': pca,
# #             'pca_cols': pca_cols  # columns PCA was applied to
# #         }, f)
# #         # pickle.dump({'model':best_model_info['model_object'], 'scaler': scaler}, f)

# #     # Output
# #     output = {
# #     'best_model': best_model_info['model_name'],
# #     'metric_name': 'Accuracy' if problem_type == 'classification' else 'R-squared',
# #     'score': best_model_info['score'],
# #     'best_params': best_model_info['best_params'],
# #     'model_path': model_path,
# #     'all_results': results
# # }

# #     print(json.dumps(output))

# # if __name__ == '__main__':
# #     main()



# this is newer 


# import sys
# import json
# import pickle
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, precision_score, recall_score, f1_score
# from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
# from sklearn.svm import SVC, SVR
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from xgboost import XGBClassifier
# from sklearn.decomposition import PCA
# from imblearn.over_sampling import SMOTE
# import warnings
# from dotenv import load_dotenv

# warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
# warnings.filterwarnings("ignore", category=FutureWarning)

# def main():
#     load_dotenv()
#     file_path = sys.argv[1]
#     target_column = sys.argv[2]
#     problem_type = sys.argv[3].lower()
#     selected_models_str = sys.argv[4]
#     selected_models = json.loads(selected_models_str)
#     replace_column = sys.argv[5]
#     find_value = sys.argv[6]
#     replace_value = sys.argv[7]

#     print(find_value, file=sys.stderr)
#     print(replace_value, file=sys.stderr)

#     if replace_value and isinstance(replace_value, str) and replace_value.lower() == 'null':
#         replace_value = None

#     df = pd.read_csv(file_path)

#     if replace_column and replace_column in df.columns:
#         print(df[replace_column].head(), file=sys.stderr)
#         df[replace_column] = df[replace_column].replace(find_value, replace_value)
#         print(df[replace_column].head(), file=sys.stderr)
#     else:
#         print(f"[Info] replace_column '{replace_column}' not provided or not found in dataset.", file=sys.stderr)

#     # if replace_value.lower() == 'null':
#     #     replace_value = None

#     # df = pd.read_csv(file_path)
#     # print(df[replace_column].head(), file=sys.stderr)

#     # if replace_column and replace_column in df.columns:
#     #     df[replace_column] = df[replace_column].replace(find_value, replace_value)

#     #     # before_values = df[replace_column].copy()

#     #     df[replace_column] = df[replace_column].replace(find_value, replace_value)
#     #     print(df[replace_column].head(), file=sys.stderr)

#     # Handle missing values
#     for col in df.columns:
#         if df[col].dtype in ['float64', 'int64']:
#             df[col] = df[col].fillna(df[col].mean())
#         else:
#             df[col] = df[col].fillna(df[col].mode()[0])

#     # Remove commas in numeric strings
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             try:
#                 df[col] = df[col].str.replace(',', '').astype(float)
#             except:
#                 pass

#     X = df.drop(columns=[target_column])
    
#     y = df[target_column]
#     X = pd.get_dummies(X, drop_first=True)
#     feature_columns = X.columns.tolist()

#     # Encode target for classification
#     if problem_type == 'classification' and y.dtype == 'object':
#         le = LabelEncoder()
#         y = le.fit_transform(y)

#     # -------------------------
#     # Train/Test Split
#     # -------------------------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=42
#     )

#     # -------------------------
#     # SMOTE only on training set
#     # -------------------------
#     # if problem_type == 'classification':
#     #     smote = SMOTE(random_state=42)
#     #     X_train, y_train = smote.fit_resample(X_train, y_train)

#     if problem_type == "classification":
    
#         # Count class distribution
#         class_counts = pd.Series(y_train).value_counts().sort_values(ascending=False)
#         max_count = class_counts.iloc[0]
#         mean_count = int(class_counts.mean())

#         imbalance_factor = max_count / (mean_count + 1e-9)
#         imbalance_threshold = 1.5  # if max/mean > this, we consider imbalanced
#         balance_mode = 'to_moderate'  # 'to_max' for full balance

#         if imbalance_factor > imbalance_threshold:
#             if balance_mode == 'to_max':
#                 target_count = int(max_count)
#             else:  # to_moderate
#                 target_count = min(int(mean_count * 1.5), int(max_count))

#             # Only balance classes below target_count
#             sampling_strategy = {int(cls): target_count for cls, cnt in class_counts.items() if cnt < target_count}

#             if sampling_strategy:
#                 smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
#                 X_train, y_train = smote.fit_resample(X_train, y_train)
#                 print(f"[SMOTE Applied] Before: {class_counts.to_dict()}", file=sys.stderr)
#                 print(f"[SMOTE Applied] After : {pd.Series(y_train).value_counts().to_dict()}", file=sys.stderr)
#             else:
#                 print("[SMOTE Skipped] No minority class below target count.", file=sys.stderr)
#         else:
#             print(f"[SMOTE Skipped] imbalance_factor={imbalance_factor:.2f} <= threshold {imbalance_threshold}", file=sys.stderr)

#     # -------------------------
#     # Scaling + PCA without leakage
#     # -------------------------
#     numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
#     continuous_cols = [
#         col for col in numeric_cols
#         if X_train[col].nunique() > 5
#     ]

#     scaler = None
#     pca = None
#     pca_cols = continuous_cols

#     if continuous_cols:
#         scaler = StandardScaler()
#         X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
#         X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

#         pca = PCA(n_components=0.95)
#         X_train_pca = pca.fit_transform(X_train[continuous_cols])
#         X_test_pca = pca.transform(X_test[continuous_cols])

#         pca_col_names = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]

#         X_train = X_train.drop(columns=continuous_cols).reset_index(drop=True)
#         X_test = X_test.drop(columns=continuous_cols).reset_index(drop=True)

#         X_train = pd.concat([X_train, pd.DataFrame(X_train_pca, columns=pca_col_names)], axis=1)
#         X_test = pd.concat([X_test, pd.DataFrame(X_test_pca, columns=pca_col_names)], axis=1)

#         print("\n[Scaling + PCA Applied]", file=sys.stderr)
#         print("Scaled columns:", list(continuous_cols), file=sys.stderr)
#         print("PCA components kept:", pca.n_components_, file=sys.stderr)
#         print("Explained variance ratio:", pca.explained_variance_ratio_, file=sys.stderr)
#         print("----------------------\n", file=sys.stderr)

#     # -------------------------
#     # Models
#     # -------------------------
#     classification_models = {
#         'Logistic Regression': LogisticRegression(max_iter=1000),
#         'Random Forest': RandomForestClassifier(random_state=42),
#         'Support Vector Machine': SVC(probability=True, random_state=42),
#         'AdaBoost': AdaBoostClassifier(random_state=42),
#         'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
#         'Decission Tree': DecisionTreeClassifier(),
#     }

#     regression_models = {
#         'Linear Regression': LinearRegression(),
#         'Ridge': Ridge(),
#         'Lasso': Lasso(),
#         'Random Forest': RandomForestRegressor(random_state=42),
#         'Support Vector Machine': SVR(),
#         'Decission Tree': DecisionTreeRegressor()
#     }

#     classification_params = {
#         'Logistic Regression': {
#             'C': [0.01, 0.1, 1, 10],
#             'solver': ['lbfgs', 'liblinear']
#         },
#         'Random Forest': {
#             'n_estimators': [100, 200],
#             'max_depth': [5, 10],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         },
#         'Support Vector Machine': {
#             'C': [0.1, 1, 10],
#             'kernel': ['linear', 'rbf'],
#             'gamma': ['scale', 'auto']
#         },
#         'AdaBoost': {
#             'n_estimators': [50, 100, 200],
#             'learning_rate': [0.01, 0.1, 1]
#         },
#         'XGBoost': {
#             'n_estimators': [100, 200],
#             'max_depth': [3, 5, 7],
#             'learning_rate': [0.01, 0.1, 0.3],
#             'subsample': [0.8, 1.0]
#         },
#         'Decision Tree': {
#             'criterion': ['gini', 'entropy'],
#             'splitter': ['best', 'random'],
#             'max_depth': [None, 10, 20, 30, 40, 50],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4],
#             'max_features': [None, 'sqrt', 'log2']
#         }
#     }

#     regression_params = {
#         'Linear Regression': {},
#         'Ridge': {
#             'alpha': [0.1, 1.0, 10.0]
#         },
#         'Lasso': {
#             'alpha': [0.1, 1.0, 10.0]
#         },
#         'Random Forest': {
#             'n_estimators': [100, 200],
#             'max_depth': [5, 10],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         },
#         'Support Vector Machine': {
#             'C': [0.1, 1, 10],
#             'kernel': ['linear', 'rbf'],
#             'gamma': ['scale', 'auto']
#         },
#         'Decision Tree': {
#             'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
#             'splitter': ['best', 'random'],
#             'max_depth': [None, 10, 20, 30, 40, 50],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4],
#             'max_features': [None, 'sqrt', 'log2']
#         }
#     }

#     if problem_type == 'classification':
#         model_map = classification_models
#         param_grids = classification_params
#         scoring_metric = 'accuracy'
#     else:
#         model_map = regression_models
#         param_grids = regression_params
#         scoring_metric = 'r2'

#     if "best_one" in selected_models:
#         selected_models = list(model_map.keys())

#     results = {}
#     best_model_info = {'model_name': None, 'score': -1, 'model_object': None, 'best_params': None}

#     for model_name in selected_models:
#         if model_name not in model_map:
#             continue

#         model = model_map[model_name]
#         params = param_grids.get(model_name, {})

#         if params:
#             grid = GridSearchCV(
#                 estimator=model,
#                 param_grid=params,
#                 cv=5,
#                 scoring=scoring_metric,
#                 n_jobs=3
#             )
#             grid.fit(X_train, y_train)
#             best_estimator = grid.best_estimator_
#             best_params = grid.best_params_
#             cv_score = grid.best_score_
#         else:
#             best_estimator = model
#             best_estimator.fit(X_train, y_train)
#             best_params = {}
#             cv_score = cross_val_score(best_estimator, X_train, y_train, cv=5, scoring=scoring_metric).mean()

#         train_score = best_estimator.score(X_train, y_train)
#         test_score = best_estimator.score(X_test, y_test)

#         generalization_gap = abs(train_score - test_score)
#         if generalization_gap > 0.1:
#             test_score -= generalization_gap * 0.5

#         if problem_type == 'classification':
#             y_pred = best_estimator.predict(X_test)
#             cm = confusion_matrix(y_test, y_pred).tolist()
#             precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#             recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
#             f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
#         else:
#             cm = None
#             precision = None
#             recall = None
#             f1 = None

#         results[model_name] = {
#             'cv_score': cv_score,
#             'train_score': train_score,
#             'test_score': test_score,
#             'generalization_gap': generalization_gap,
#             'best_params': best_params,
#             'confusion_matrix': cm,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1
#         }

#         if test_score > best_model_info['score']:
#             best_model_info = {
#                 'model_name': model_name,
#                 'score': test_score,
#                 'model_object': best_estimator,
#                 'best_params': best_params
#             }

#     model_filename = f'model_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
#     model_path = f'uploads/{model_filename}'
#     with open(model_path, 'wb') as f:
#         pickle.dump({
#             'model': best_model_info['model_object'],
#             'scaler': scaler,
#             'pca': pca,
#             'pca_cols': pca_cols,
#             'feature_columns':feature_columns,
#         }, f)

#     output = {
#         'best_model': best_model_info['model_name'],
#         'metric_name': 'Accuracy' if problem_type == 'classification' else 'R-squared',
#         'score': best_model_info['score'],
#         'best_params': best_model_info['best_params'],
#         'model_path': model_path,
#         'all_results': results
#     }

#     print(json.dumps(output))

# if __name__ == '__main__':
#     main()


import sys
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    load_dotenv()

    # Read arguments
    file_path = sys.argv[1]
    target_column = sys.argv[2]
    problem_type = sys.argv[3].lower()
    selected_models_str = sys.argv[4]
    selected_models = json.loads(selected_models_str)
    replace_column = sys.argv[5]
    find_value = sys.argv[6]
    replace_value = sys.argv[7]

    scaler = None
    pca = None
    pca_cols = []
    # Environment-controlled values
    test_size = float(os.getenv("TRAIN_TEST_SPLIT", "0.25"))
    random_state = int(os.getenv("RANDOM_STATE", "42"))
    smote_balance_mode = os.getenv("SMOTE_BALANCE_MODE", "to_moderate")  # to_moderate or to_max
    smote_threshold = float(os.getenv("SMOTE_IMBALANCE_THRESHOLD", "1.5"))
    n_jobs = int(os.getenv("GRIDSEARCH_NJOBS", "3"))
    uploads_dir = os.getenv("UPLOADS_DIR", "uploads")

    print(find_value, file=sys.stderr)
    print(replace_value, file=sys.stderr)

    if replace_value and isinstance(replace_value, str) and replace_value.lower() == 'null':
        replace_value = None

    df = pd.read_csv(file_path)

    if replace_column and replace_column in df.columns:
        print(df[replace_column].head(), file=sys.stderr)
        df[replace_column] = df[replace_column].replace(find_value, replace_value)
        print(df[replace_column].head(), file=sys.stderr)
    else:
        print(f"[Info] replace_column '{replace_column}' not provided or not found in dataset.", file=sys.stderr)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove commas in numeric strings
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '').astype(float)
            except:
                pass

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = pd.get_dummies(X, drop_first=True)
    feature_columns = X.columns.tolist()

    # Encode target for classification
    if problem_type == 'classification' and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Handle class imbalance with SMOTE
    if problem_type == "classification":
        class_counts = pd.Series(y_train).value_counts().sort_values(ascending=False)
        max_count = class_counts.iloc[0]
        mean_count = int(class_counts.mean())

        imbalance_factor = max_count / (mean_count + 1e-9)

        if imbalance_factor > smote_threshold:
            if smote_balance_mode == 'to_max':
                target_count = int(max_count)
            else:
                target_count = min(int(mean_count * 1.5), int(max_count))

            sampling_strategy = {int(cls): target_count for cls, cnt in class_counts.items() if cnt < target_count}

            if sampling_strategy:
                smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"[SMOTE Applied] Before: {class_counts.to_dict()}", file=sys.stderr)
                print(f"[SMOTE Applied] After : {pd.Series(y_train).value_counts().to_dict()}", file=sys.stderr)
            else:
                print("[SMOTE Skipped] No minority class below target count.", file=sys.stderr)
        else:
            print(f"[SMOTE Skipped] imbalance_factor={imbalance_factor:.2f} <= threshold {smote_threshold}", file=sys.stderr)

    # Scaling + PCA
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    continuous_cols = [col for col in numeric_cols if X_train[col].nunique() > 5]

    scaler = None
    pca = None
    pca_cols = continuous_cols

    if continuous_cols:
        scaler = StandardScaler()
        X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
        X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train[continuous_cols])
        X_test_pca = pca.transform(X_test[continuous_cols])

        pca_col_names = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]

        X_train = X_train.drop(columns=continuous_cols).reset_index(drop=True)
        X_test = X_test.drop(columns=continuous_cols).reset_index(drop=True)

        X_train = pd.concat([X_train, pd.DataFrame(X_train_pca, columns=pca_col_names)], axis=1)
        X_test = pd.concat([X_test, pd.DataFrame(X_test_pca, columns=pca_col_names)], axis=1)

        print("\n[Scaling + PCA Applied]", file=sys.stderr)
        print("Scaled columns:", list(continuous_cols), file=sys.stderr)
        print("PCA components kept:", pca.n_components_, file=sys.stderr)
        print("Explained variance ratio:", pca.explained_variance_ratio_, file=sys.stderr)
        print("----------------------\n", file=sys.stderr)

    # Define models & params
    classification_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Support Vector Machine': SVC(probability=True, random_state=random_state),
        'AdaBoost': AdaBoostClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
    }

    regression_models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'Support Vector Machine': SVR(),
        'Decision Tree': DecisionTreeRegressor(random_state=random_state)
    }

    classification_params = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Support Vector Machine': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        },
        'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }
    }

    regression_params = {
        'Linear Regression': {},
        'Ridge': {
            'alpha': [0.1, 1.0, 10.0]
        },
        'Lasso': {
            'alpha': [0.1, 1.0, 10.0]
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Support Vector Machine': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'Decision Tree': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }
    }

    if problem_type == 'classification':
        model_map = classification_models
        param_grids = classification_params
        scoring_metric = 'accuracy'
    else:
        model_map = regression_models
        param_grids = regression_params
        scoring_metric = 'r2'

    if "best_one" in selected_models:
        selected_models = list(model_map.keys())

    results = {}
    best_model_info = {'model_name': None, 'score': -1, 'model_object': None, 'best_params': None}

    for model_name in selected_models:
        if model_name not in model_map:
            continue

        model = model_map[model_name]
        params = param_grids.get(model_name, {})

        if params:
            grid = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=5,
                scoring=scoring_metric,
                n_jobs=n_jobs
            )
            grid.fit(X_train, y_train)
            best_estimator = grid.best_estimator_
            best_params = grid.best_params_
            cv_score = grid.best_score_
        else:
            best_estimator = model
            best_estimator.fit(X_train, y_train)
            best_params = {}
            cv = min(5, len(X_train))
            if cv < 2:
                cv = 2
            cv_score = cross_val_score(best_estimator, X_train, y_train, cv=5, scoring=scoring_metric).mean()

        train_score = best_estimator.score(X_train, y_train)
        test_score = best_estimator.score(X_test, y_test)

        generalization_gap = abs(train_score - test_score)
        if generalization_gap > 0.1:
            test_score -= generalization_gap * 0.5

        le = None

        if problem_type == 'classification':
            y_pred = best_estimator.predict(X_test)
            if le is not None:
                y_pred_labels = le.inverse_transform(y_pred)
                y_test_labels = le.inverse_transform(y_test)
            else:
                y_pred_labels = y_pred
                y_test_labels = y_test
            cm = confusion_matrix(y_test, y_pred).tolist()
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        else:
            cm = None
            precision = None
            recall = None
            f1 = None
            y_pred_labels = None
            y_test_labels = None

        results[model_name] = {
            'cv_score': cv_score,
            'train_score': train_score,
            'test_score': test_score,
            'generalization_gap': generalization_gap,
            'best_params': best_params,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred_labels.tolist() if y_pred_labels is not None else None,
            'y_test': y_test_labels.tolist() if y_test_labels is not None else None
        }

        if test_score > best_model_info['score']:
            best_model_info = {
                'model_name': model_name,
                'score': test_score,
                'model_object': best_estimator,
                'best_params': best_params
            }

    # Save model to configured uploads dir
    model_filename = f'model_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    os.makedirs(uploads_dir, exist_ok=True)
    model_path = os.path.join(uploads_dir, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_model_info['model_object'],
            'scaler': scaler,
            'pca': pca,
            'pca_cols': pca_cols,
            'feature_columns': feature_columns,
            'label_encoder': le
        }, f)

    output = {
        'best_model': best_model_info['model_name'],
        'metric_name': 'Accuracy' if problem_type == 'classification' else 'R-squared',
        'score': best_model_info['score'],
        'best_params': best_model_info['best_params'],
        'model_path': model_path,
        'all_results': results
    }

    print(json.dumps(output))

if __name__ == '__main__':
    main()
