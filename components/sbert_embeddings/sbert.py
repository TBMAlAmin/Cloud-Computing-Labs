import argparse
import os
import glob
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_data", required=True)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.input_data, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet file found in {args.input_data}")

    df = pd.read_parquet(files[0])

    if "reviewText" not in df.columns:
        raise ValueError("Expected column 'reviewText' not found")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["reviewText"].astype(str).tolist()

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    emb_df = pd.DataFrame(embeddings)
    emb_df.columns = emb_df.columns.map(str)  # parquet-safe

    os.makedirs(args.output_data, exist_ok=True)
    emb_df.to_parquet(os.path.join(args.output_data, "sbert_embeddings.parquet"), index=False)

    print("✅ sbert_embeddings completed")

if __name__ == "__main__":
    main()
