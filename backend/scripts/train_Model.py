# import sys
# import json
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report
# from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge,Lasso
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.svm import SVC, SVR
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
# warnings.filterwarnings("ignore", category=FutureWarning)

# def main():
#     file_path = sys.argv[1]
#     target_column = sys.argv[2]
#     problem_type = sys.argv[3]  # "classification" or "regression"
#     selected_models_str = sys.argv[4]
#     selected_models = json.loads(selected_models_str)

#     # Load data
#     df = pd.read_csv(file_path)

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

#     # One-hot encoding for categorical features
#     X = pd.get_dummies(X, drop_first=True)

#     # Encode target if classification
#     if problem_type == 'classification':
#         if y.dtype == 'object':
#             le = LabelEncoder()
#             y = le.fit_transform(y)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=42
#     )

#     # Model definitions
#     model_map = {
#         'Linear Regression': LinearRegression(),
#         'Logistic Regression': LogisticRegression(max_iter=1000),
#         'Ridge Regression': Ridge(),
#         'Lasso Regression': Lasso(),
#         'Random Forest': RandomForestClassifier(random_state=42) if problem_type == 'classification' else RandomForestRegressor(random_state=42),
#         'Support Vector Machine': SVC(probability=True, random_state=42) if problem_type == 'classification' else SVR()
#     }

#     # Parameter grids for tuning
#     param_grids = {
#         'Linear Regression': {},# no hyperparams
#         'Ridge Regression': {
#             'alpha': [0.1, 1.0, 10.0, 100.0]
#         },
#         'Lasso Regression': {
#             'alpha': [0.1, 1.0, 10.0, 100.0]
#         },
#         'Logistic Regression': {
#             'C': [0.01, 0.1, 1, 10],
#             'solver': ['lbfgs', 'liblinear']
#         },
#         'Random Forest': {
#             'n_estimators': [100, 200, 300],
#             'max_depth': [5, 10, None],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4]
#         },
#         'Support Vector Machine': {
#             'C': [0.1, 1, 10],
#             'kernel': ['linear', 'rbf'],
#             'gamma': ['scale', 'auto']
#         }
#     }

#     # If "Best" is selected → try all models
#     if "best_one" in selected_models:
#         selected_models = list(model_map.keys())

#     results = {}
#     best_model_info = {'model_name': None, 'score': -1, 'model_object': None, 'best_params': None}

#     # Train/tune each selected model
#     for model_name in selected_models:
#         if model_name not in model_map:
#             continue

#         model = model_map[model_name]
#         params = param_grids.get(model_name, {})

#         if params:  # If model has parameters to tune
#             grid = GridSearchCV(
#                 estimator=model,
#                 param_grid=params,
#                 cv=5,
#                 scoring='accuracy' if problem_type == 'classification' else 'r2',
#                 n_jobs=-1
#             )
#             grid.fit(X_train, y_train)
#             best_estimator = grid.best_estimator_
#             best_params = grid.best_params_
#         else:  # No params to tune
#             model.fit(X_train, y_train)
#             best_estimator = model
#             best_params = {}

#         predictions = best_estimator.predict(X_test)

#         if problem_type == 'classification':
#             score = accuracy_score(y_test, predictions)
#             cf = confusion_matrix(y_test, predictions)
#             report = classification_report(y_test, predictions)
#         else:
#             score = r2_score(y_test, predictions)

#         results[model_name] = {
#             'score': score,
#             'best_params': best_params
#         }

#         if score > best_model_info['score']:
#             best_model_info = {
#                 'model_name': model_name,
#                 'score': score,
#                 'model_object': best_estimator,
#                 'best_params': best_params
#             }

#     # Save best model
#     model_filename = f'model_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
#     model_path = f'uploads/{model_filename}'
#     with open(model_path, 'wb') as f:
#         pickle.dump(best_model_info['model_object'], f)

#     # Output results
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    file_path = sys.argv[1]
    target_column = sys.argv[2]
    problem_type = sys.argv[3].lower()
    selected_models_str = sys.argv[4]
    selected_models = json.loads(selected_models_str)

    df = pd.read_csv(file_path)

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

    # Encode target for classification
    if problem_type == 'classification' and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Model dictionaries
    classification_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }

    regression_models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Support Vector Machine': SVR()
    }

    # Parameter grids with limits to avoid overfitting
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
        }
    }

    # Choose correct model set
    if problem_type == 'classification':
        model_map = classification_models
        param_grids = classification_params
        scoring_metric = 'accuracy'
    else:
        model_map = regression_models
        param_grids = regression_params
        scoring_metric = 'r2'

    # If best_one selected, run all models of correct type
    if "best_one" in selected_models:
        selected_models = list(model_map.keys())

    results = {}
    best_model_info = {'model_name': None, 'score': -1, 'model_object': None, 'best_params': None}

    # Train models
    for model_name in selected_models:
        if model_name not in model_map:
            continue

        model = model_map[model_name]
        params = param_grids.get(model_name, {})

        # Hyperparameter tuning with CV
        if params:
            grid = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=5,
                scoring=scoring_metric,
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_estimator = grid.best_estimator_
            best_params = grid.best_params_
            cv_score = grid.best_score_
        else:
            best_estimator = model
            best_estimator.fit(X_train, y_train)
            best_params = {}
            cv_score = cross_val_score(best_estimator, X_train, y_train, cv=5, scoring=scoring_metric).mean()

        # Evaluate
        train_score = best_estimator.score(X_train, y_train)
        test_score = best_estimator.score(X_test, y_test)

        # Overfitting check
        generalization_gap = abs(train_score - test_score)
        if generalization_gap > 0.1:
            test_score -= generalization_gap * 0.5  # penalize overfit models

        results[model_name] = {
            'cv_score': cv_score,
            'train_score': train_score,
            'test_score': test_score,
            'generalization_gap': generalization_gap,
            'best_params': best_params
        }

        if test_score > best_model_info['score']:
            best_model_info = {
                'model_name': model_name,
                'score': test_score,
                'model_object': best_estimator,
                'best_params': best_params
            }

    # Save best model
    model_filename = f'model_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    model_path = f'uploads/{model_filename}'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_info['model_object'], f)

    # Output
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
