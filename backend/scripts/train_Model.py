import sys
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    load_dotenv()

    # ---- Read CLI arguments ----
    file_path = sys.argv[1]
    target_column = sys.argv[2]
    problem_type = sys.argv[3].lower()
    selected_models_str = sys.argv[4]
    selected_models = json.loads(selected_models_str)
    cleaning_rules = json.loads(sys.argv[5])

    # ---- Env config ----
    test_size = float(os.getenv("TRAIN_TEST_SPLIT", "0.25"))
    random_state = int(os.getenv("RANDOM_STATE", "42"))
    smote_balance_mode = os.getenv("SMOTE_BALANCE_MODE", "to_moderate")  # to_moderate or to_max
    smote_threshold = float(os.getenv("SMOTE_IMBALANCE_THRESHOLD", "1.5"))
    n_jobs = int(os.getenv("GRIDSEARCH_NJOBS", "3"))
    uploads_dir = os.getenv("UPLOADS_DIR", "uploads")

    # ---- Ensure optionally assigned objects are defined ----
    le = None
    scaler = None
    pca = None
    pca_cols = []

    # ---- Load data ----
    df = pd.read_csv(file_path)

    # ---- Apply cleaning rules (single pass, safe) ----
    if isinstance(cleaning_rules, list) and cleaning_rules:
        for rule in cleaning_rules:
            replace_column = rule.get("replaceColumn")
            find_value = rule.get("findValue")
            replace_value = rule.get("replaceValue")
            sys.stderr.write(f"Before Cleaning Rule: {df[replace_column].head()}\n")

            # Handle "null" string as missing
            if isinstance(replace_value, str) and replace_value.lower() == "null":
                replace_value = pd.NA

            if replace_column and replace_column in df.columns:
                sys.stderr.write(f"[Replace] {find_value} → {replace_value} in column {replace_column}\n")
                df[replace_column] = df[replace_column].replace(find_value, replace_value)
            else:
                sys.stderr.write(f"[Skip] Column {replace_column} not found in dataset.\n")
            sys.stderr.write(f"After Cleaning Rule: {df[replace_column].head()}\n")
    else:
        sys.stderr.write("[Info] No cleaning rules provided.\n")

    # ---- Handle missing values ----
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            # guard against empty mode
            mode_vals = df[col].mode(dropna=True)
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals.iloc[0])
            else:
                df[col] = df[col].fillna("")
                
    # ---- Remove commas in numeric-looking object columns ----
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # only attempt if most values look numeric after comma removal
                sample = df[col].dropna().astype(str).str.replace(',', '', regex=False)
                can_cast_ratio = (sample.str.fullmatch(r"-?\d+(\.\d+)?")).mean()
                if can_cast_ratio >= 0.9:
                    df[col] = sample.astype(float)
            except Exception:
                pass

    # ---- Split features/target ----
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # One-hot encode features
    X = pd.get_dummies(X, drop_first=True)
    feature_columns = X.columns.tolist()

    # ---- Encode target for classification if needed ----
    if problem_type == 'classification':
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        # else: assume already numeric labels, keep le=None

    # ---- Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ---- SMOTE for imbalanced classification ----
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

            sampling_strategy = {
                int(cls): target_count
                for cls, cnt in class_counts.items()
                if cnt < target_count
            }

            if sampling_strategy:
                smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                sys.stderr.write(f"[SMOTE Applied] Before: {class_counts.to_dict()}\n")
                sys.stderr.write(f"[SMOTE Applied] After : {pd.Series(y_train).value_counts().to_dict()}\n")
            else:
                sys.stderr.write("[SMOTE Skipped] No minority class below target count.\n")
        else:
            sys.stderr.write(f"[SMOTE Skipped] imbalance_factor={imbalance_factor:.2f} <= threshold {smote_threshold}\n")

    # ---- Scaling + PCA on continuous numeric features ----
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    continuous_cols = [col for col in numeric_cols if X_train[col].nunique() > 5]
    pca_cols = continuous_cols[:]  # keep track for inference

    if continuous_cols:
        scaler = StandardScaler()
        X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
        X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train[continuous_cols])
        X_test_pca = pca.transform(X_test[continuous_cols])

        pca_col_names = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]

        # Drop original continuous cols and append PCs
        X_train = X_train.drop(columns=continuous_cols).reset_index(drop=True)
        X_test = X_test.drop(columns=continuous_cols).reset_index(drop=True)
        X_train = pd.concat([X_train, pd.DataFrame(X_train_pca, columns=pca_col_names)], axis=1)
        X_test = pd.concat([X_test, pd.DataFrame(X_test_pca, columns=pca_col_names)], axis=1)

        sys.stderr.write("\n[Scaling + PCA Applied]\n")
        sys.stderr.write(f"Scaled columns: {list(continuous_cols)}\n")
        sys.stderr.write(f"PCA components kept: {pca.n_components_}\n")
        sys.stderr.write(f"Explained variance ratio: {pca.explained_variance_ratio_}\n")
        sys.stderr.write("----------------------\n\n")

    # ---- Models & params ----
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
        'Ridge': {'alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'alpha': [0.1, 1.0, 10.0]},
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

    # ---- Train/evaluate ----
    results = {}
    best_model_info = {'model_name': None, 'score': -1, 'model_object': None, 'best_params': None}

    for model_name in selected_models:
        if model_name not in model_map:
            continue

        model = model_map[model_name]
        params = param_grids.get(model_name, {})


        if params:

            rand = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=10,             # try only 10 random combos
                cv=3,                  # also reduce folds
                scoring=scoring_metric,
                n_jobs=1,              # use 1 to save memory on Render free plan
                random_state=42
            )
            rand.fit(X_train, y_train)
            best_estimator = rand.best_estimator_
            best_params = rand.best_params_
            cv_score = rand.best_score_

            # grid = GridSearchCV(
            #     estimator=model,
            #     param_grid=params,
            #     cv=5,
            #     scoring=scoring_metric,
            #     n_jobs=n_jobs
            # )
            # grid.fit(X_train, y_train)
            # best_estimator = grid.best_estimator_
            # best_params = grid.best_params_
            # cv_score = rand.best_score_
        else:
            # Fit once for cv, then refit on all train (GridSearchCV would refit automatically)
            cv_score = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring_metric, n_jobs=n_jobs).mean()
            best_estimator = model
            best_estimator.fit(X_train, y_train)
            best_params = {}

        train_score = best_estimator.score(X_train, y_train)
        test_score = best_estimator.score(X_test, y_test)

        generalization_gap = abs(train_score - test_score)
        if generalization_gap > 0.1:
            # penalize potential overfitting for ranking (does not change raw metrics we display)
            test_score -= generalization_gap * 0.5

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

    # ---- Save best model bundle ----
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

    # ---- Output summary JSON ----
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

