# import sys
# import json
# import pickle
# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv

# def main():
#     load_dotenv()
#     model_path = sys.argv[1]
#     # Read JSON input from stdin instead of argv
#     input_json = sys.stdin.read()

#     print(model_path, file=sys.stderr)
#     print(input_json, file=sys.stderr)

#     # Load the model dict from pickle
#     with open(model_path, 'rb') as f:
#         data = pickle.load(f)

#     model = data['model']
#     scaler = data.get('scaler')
#     pca = data.get('pca')
#     pca_cols = data.get('pca_cols')
#     feature_columns = data.get('feature_columns')

#     input_data = json.loads(input_json)
#     df = pd.DataFrame([input_data])

#     for col in df.columns:
#         if df[col].dtype == object:
#             try:
#                 df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
#             except:
#                 pass

#     df = pd.get_dummies(df, drop_first=True)

#     if feature_columns is not None:
#         missing_cols = set(feature_columns) - set(df.columns)
#         for c in missing_cols:
#             df[c] = 0
#         df = df[feature_columns]

#     if scaler and pca and pca_cols:
#         df_scaled = df.copy()
#         df_scaled[pca_cols] = scaler.transform(df[pca_cols])
#         pca_result = pca.transform(df_scaled[pca_cols])
#         pca_col_names = [f'PC{i+1}' for i in range(pca_result.shape[1])]
#         df_scaled = df_scaled.drop(columns=pca_cols)
#         df_scaled[pca_col_names] = pca_result
#         df = df_scaled

#     pred = model.predict(df)

#     if hasattr(pred, '__len__'):
#         pred = pred.tolist()

#     print(json.dumps({'prediction': pred}))

# if __name__ == '__main__':
#     main()


# import sys
# import os
# import json
# import pickle
# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv

# def main():
#     # Load .env
#     load_dotenv()

#     # ==============================
#     # Get model path (CLI > .env > default)
#     # ==============================
#     if len(sys.argv) > 1:
#         model_path = sys.argv[1]
#     else:
#         model_path = os.getenv("MODEL_PATH", "./uploads/default_model.pkl")

#     # ==============================
#     # Read JSON input from stdin
#     # ==============================
#     input_json = sys.stdin.read().strip()
#     if not input_json:
#         input_json = os.getenv("INPUT_JSON", "{}")  # fallback for testing

#     print(f"🔹 Using model: {model_path}", file=sys.stderr)
#     print(f"🔹 Input JSON: {input_json}", file=sys.stderr)

#     # ==============================
#     # Load model
#     # ==============================
#     try:
#         with open(model_path, 'rb') as f:
#             data = pickle.load(f)
#     except Exception as e:
#         print(json.dumps({"error": f"Failed to load model: {str(e)}"}))
#         sys.exit(1)

#     model = data['model']
#     scaler = data.get('scaler')
#     pca = data.get('pca')
#     pca_cols = data.get('pca_cols')
#     feature_columns = data.get('feature_columns')
#     label_encoder = data.get('label_encoder', None)

#     # ==============================
#     # Prepare input
#     # ==============================
#     try:
#         input_data = json.loads(input_json)
#         df = pd.DataFrame([input_data])

#         # Convert string numerics
#         for col in df.columns:
#             if df[col].dtype == object:
#                 try:
#                     df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
#                 except Exception:
#                     pass

#         # One-hot encode
#         df = pd.get_dummies(df, drop_first=True)

#         # Align with training columns
#         if feature_columns is not None:
#             missing_cols = set(feature_columns) - set(df.columns)
#             for c in missing_cols:
#                 df[c] = 0
#             df = df[feature_columns]

#         # Apply scaler + PCA if available
#         if scaler and pca and pca_cols:
#             df_scaled = df.copy()
#             df_scaled[pca_cols] = scaler.transform(df[pca_cols])
#             pca_result = pca.transform(df_scaled[pca_cols])
#             pca_col_names = [f'PC{i+1}' for i in range(pca_result.shape[1])]
#             df_scaled = df_scaled.drop(columns=pca_cols)
#             df_scaled[pca_col_names] = pca_result
#             df = df_scaled

#         # ==============================
#         # Make prediction
#         # ==============================
#         pred = model.predict(df)

#         if hasattr(pred, '__len__'):
#             pred = pred.tolist()

#         print(json.dumps({'prediction': pred}))

#     except Exception as e:
#         print(json.dumps({"error": f"Prediction failed: {str(e)}"}))
#         sys.exit(1)

# if __name__ == '__main__':
#     main()



import sys
import os
import json
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv

def main():
    # Load .env
    load_dotenv()

    # ==============================
    # Get model path (CLI > .env > default)
    # ==============================
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.getenv("MODEL_PATH", "./uploads/default_model.pkl")

    # ==============================
    # Read JSON input from stdin
    # ==============================
    input_json = sys.stdin.read().strip()
    if not input_json:
        input_json = os.getenv("INPUT_JSON", "{}")  # fallback for testing

    print(f"🔹 Using model: {model_path}", file=sys.stderr)
    print(f"🔹 Input JSON: {input_json}", file=sys.stderr)

    # ==============================
    # Load model
    # ==============================
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {str(e)}"}))
        sys.exit(1)

    model = data['model']
    scaler = data.get('scaler')
    pca = data.get('pca')
    pca_cols = data.get('pca_cols')
    feature_columns = data.get('feature_columns')
    label_encoder = data.get('label_encoder', None)   # ✅ added

    # ==============================
    # Prepare input
    # ==============================
    try:
        input_data = json.loads(input_json)
        df = pd.DataFrame([input_data])

        # Convert string numerics
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
                except Exception:
                    pass

        # One-hot encode
        df = pd.get_dummies(df, drop_first=True)

        # Align with training columns
        if feature_columns is not None:
            missing_cols = set(feature_columns) - set(df.columns)
            for c in missing_cols:
                df[c] = 0
            df = df[feature_columns]

        # Apply scaler + PCA if available
        if scaler and pca and pca_cols:
            df_scaled = df.copy()
            df_scaled[pca_cols] = scaler.transform(df[pca_cols])
            pca_result = pca.transform(df_scaled[pca_cols])
            pca_col_names = [f'PC{i+1}' for i in range(pca_result.shape[1])]
            df_scaled = df_scaled.drop(columns=pca_cols)
            df_scaled[pca_col_names] = pca_result
            df = df_scaled

        # ==============================
        # Make prediction
        # ==============================
        pred = model.predict(df)

        # ✅ Convert predictions back to original labels if label encoder exists
        if label_encoder is not None:
            try:
                pred = label_encoder.inverse_transform(pred)
            except Exception:
                pass

        if hasattr(pred, '__len__'):
            pred = pred.tolist()

        print(json.dumps({'prediction': pred}))

    except Exception as e:
        print(json.dumps({"error": f"Prediction failed: {str(e)}"}))
        sys.exit(1)

if __name__ == '__main__':
    main()
