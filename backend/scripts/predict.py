import sys
import json
import pickle
import pandas as pd
import numpy as np

def main():
    model_path = sys.argv[1]
    # Read JSON input from stdin instead of argv
    input_json = sys.stdin.read()

    print(model_path, file=sys.stderr)
    print(input_json, file=sys.stderr)

    # Load the model dict from pickle
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    scaler = data.get('scaler')
    pca = data.get('pca')
    pca_cols = data.get('pca_cols')
    feature_columns = data.get('feature_columns')

    input_data = json.loads(input_json)
    df = pd.DataFrame([input_data])

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
            except:
                pass

    df = pd.get_dummies(df, drop_first=True)

    if feature_columns is not None:
        missing_cols = set(feature_columns) - set(df.columns)
        for c in missing_cols:
            df[c] = 0
        df = df[feature_columns]

    if scaler and pca and pca_cols:
        df_scaled = df.copy()
        df_scaled[pca_cols] = scaler.transform(df[pca_cols])
        pca_result = pca.transform(df_scaled[pca_cols])
        pca_col_names = [f'PC{i+1}' for i in range(pca_result.shape[1])]
        df_scaled = df_scaled.drop(columns=pca_cols)
        df_scaled[pca_col_names] = pca_result
        df = df_scaled

    pred = model.predict(df)

    if hasattr(pred, '__len__'):
        pred = pred.tolist()

    print(json.dumps({'prediction': pred}))

if __name__ == '__main__':
    main()