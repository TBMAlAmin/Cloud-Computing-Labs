import argparse
import os
import glob
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


def find_parquet_file(folder_path: str) -> str:
    """AzureML uri_folder often contains one parquet/csv file inside. We pick the first parquet found."""
    parquet_files = glob.glob(os.path.join(folder_path, "**", "*.parquet"), recursive=True)
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet found inside folder: {folder_path}")
    return parquet_files[0]


def load_text_series(data_folder: str, text_col_candidates=("reviewText", "text", "normalized_text")) -> pd.Series:
    parquet_path = find_parquet_file(data_folder)
    df = pd.read_parquet(parquet_path)

    # pick the first matching text column
    for c in text_col_candidates:
        if c in df.columns:
            s = df[c].fillna("").astype(str)
            return s

    # fallback: if exactly one object column exists, use it
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if len(obj_cols) == 1:
        return df[obj_cols[0]].fillna("").astype(str)

    raise KeyError(
        f"Could not find a text column in {parquet_path}. "
        f"Tried {text_col_candidates} and object-column fallback."
    )


def save_dense_parquet(X, feature_names, out_dir: str, filename: str):
    """
    PyArrow cannot store pandas SparseDtype columns reliably.
    Convert to a dense numeric matrix and write as normal parquet.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert sparse -> dense numpy array
    # Use float32 to reduce memory.
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # Ensure column names are strings (parquet requirement)
    cols = [str(c) for c in feature_names]
    df_out = pd.DataFrame(X, columns=cols)

    out_path = os.path.join(out_dir, filename)
    df_out.to_parquet(out_path, index=False)
    print(f"✅ Saved: {out_path}  shape={df_out.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="uri_folder for train parquet")
    parser.add_argument("--val", type=str, required=True, help="uri_folder for val parquet")
    parser.add_argument("--test", type=str, required=True, help="uri_folder for test parquet")

    parser.add_argument("--train_out", type=str, required=True, help="output uri_folder for train tfidf parquet")
    parser.add_argument("--val_out", type=str, required=True, help="output uri_folder for val tfidf parquet")
    parser.add_argument("--test_out", type=str, required=True, help="output uri_folder for test tfidf parquet")

    # Keep features bounded to prevent OOM on DS3_v2
    parser.add_argument("--max_features", type=int, default=2000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--ngram_max", type=int, default=1)

    args = parser.parse_args()

    train_text = load_text_series(args.train)
    val_text = load_text_series(args.val)
    test_text = load_text_series(args.test)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
        dtype=np.float32
    )

    X_train = vectorizer.fit_transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)

    feature_names = vectorizer.get_feature_names_out()

    # Save as dense parquet to avoid SparseDtype Parquet failure
    save_dense_parquet(X_train, feature_names, args.train_out, "tfidf_train.parquet")
    save_dense_parquet(X_val, feature_names, args.val_out, "tfidf_val.parquet")
    save_dense_parquet(X_test, feature_names, args.test_out, "tfidf_test.parquet")


if __name__ == "__main__":
    main()