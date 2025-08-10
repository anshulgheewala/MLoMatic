


import sys
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from xgboost import XGBClassifier

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
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss',  random_state=42),
        'Decission Tree': DecisionTreeClassifier(),
    }

    regression_models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Support Vector Machine': SVR(),
        'Decission Tree': DecisionTreeRegressor()
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
        ,
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
                n_jobs=3
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
        
        # Additional metrics for classification
        if problem_type == 'classification':
            y_pred = best_estimator.predict(X_test)
            cm = confusion_matrix(y_test, y_pred).tolist()  # convert to list for JSON
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        else:
            cm = None
            precision = None
            recall = None
            f1 = None

        results[model_name] = {
            'cv_score': cv_score,
            'train_score': train_score,
            'test_score': test_score,
            'generalization_gap': generalization_gap,
            'best_params': best_params,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
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

# import sys
# import json
# import pickle
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, r2_score
# from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.svm import SVC, SVR
# import warnings

# warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
# warnings.filterwarnings("ignore", category=FutureWarning)

# def main():
#     file_path = sys.argv[1]
#     target_column = sys.argv[2]
#     problem_type = sys.argv[3].lower()
#     selected_models_str = sys.argv[4]
#     selected_models = json.loads(selected_models_str)

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
#     X = pd.get_dummies(X, drop_first=True)

#     # Encode target for classification
#     if problem_type == 'classification' and y.dtype == 'object':
#         le = LabelEncoder()
#         y = le.fit_transform(y)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=42
#     )

#     # Model dictionaries
#     classification_models = {
#         'Logistic Regression': LogisticRegression(max_iter=1000),
#         'Random Forest': RandomForestClassifier(random_state=42),
#         'Support Vector Machine': SVC(probability=True, random_state=42)
#     }

#     regression_models = {
#         'Linear Regression': LinearRegression(),
#         'Ridge': Ridge(),
#         'Lasso': Lasso(),
#         'Random Forest': RandomForestRegressor(random_state=42),
#         'Support Vector Machine': SVR()
#     }

#     # Parameter grids with limits to avoid overfitting
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
#         }
#     }

#     # Choose correct model set
#     if problem_type == 'classification':
#         model_map = classification_models
#         param_grids = classification_params
#         scoring_metric = 'accuracy'
#     else:
#         model_map = regression_models
#         param_grids = regression_params
#         scoring_metric = 'r2'

#     # If best_one selected, run all models of correct type
#     if "best_one" in selected_models:
#         selected_models = list(model_map.keys())

#     results = {}
#     best_model_info = {'model_name': None, 'score': -1, 'model_object': None, 'best_params': None}

#     # Train models
#     for model_name in selected_models:
#         if model_name not in model_map:
#             continue

#         model = model_map[model_name]
#         params = param_grids.get(model_name, {})

#         # Hyperparameter tuning with CV
#         if params:
#             grid = GridSearchCV(
#                 estimator=model,
#                 param_grid=params,
#                 cv=5,
#                 scoring=scoring_metric,
#                 n_jobs=-1
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

#         # Evaluate
#         train_score = best_estimator.score(X_train, y_train)
#         test_score = best_estimator.score(X_test, y_test)

#         # Overfitting check
#         generalization_gap = abs(train_score - test_score)
#         if generalization_gap > 0.1:
#             test_score -= generalization_gap * 0.5  # penalize overfit models

#         results[model_name] = {
#             'cv_score': cv_score,
#             'train_score': train_score,
#             'test_score': test_score,
#             'generalization_gap': generalization_gap,
#             'best_params': best_params
#         }

#         if test_score > best_model_info['score']:
#             best_model_info = {
#                 'model_name': model_name,
#                 'score': test_score,
#                 'model_object': best_estimator,
#                 'best_params': best_params
#             }

#     # Save best model
#     model_filename = f'model_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
#     model_path = f'uploads/{model_filename}'
#     with open(model_path, 'wb') as f:
#         pickle.dump(best_model_info['model_object'], f)

#     # Output
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
