#!/usr/bin/env python3
import glob
import json
import os
from pathlib import Path

import click
import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer


def read_md(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, max_tokens=300, sep="\n\n"):
    paras = text.split(sep)
    chunks, cur = [], []
    count = 0
    for p in paras:
        words = p.split()
        if count + len(words) > max_tokens and cur:
            chunks.append(sep.join(cur))
            cur, count = [], 0
        cur.append(p)
        count += len(words)
    if cur:
        chunks.append(sep.join(cur))
    return chunks


@click.command()
@click.option("--input-dir", required=True, help="folder with .md files")
@click.option("--output-dir", required=True, help="where to write shards and index")
@click.option(
    "--model-name",
    default="all-MiniLM-L6-v2",
    help="SentenceTransformer embedding model",
)
@click.option(
    "--shard-size",
    type=int,
    default=100_000,
    help="chunks per Parquet shard",
)
@click.option(
    "--max-tokens",
    type=int,
    default=500,
    help="approx tokens per chunk",
)
def main(input_dir, output_dir, model_name, shard_size, max_tokens):
    print("Loading...")
    model = SentenceTransformer(model_name)
    md_files = glob.glob(os.path.join(input_dir, "**/*.md"), recursive=True)

    # Prepare output dirs
    os.makedirs(output_dir, exist_ok=True)
    shards_dir = Path(output_dir) / "shards"
    idx_dir = Path(output_dir) / "faiss"
    shards_dir.mkdir(exist_ok=True)
    idx_dir.mkdir(exist_ok=True)

    manifest = {"parquet_shards": [], "faiss_index": str(idx_dir / "index.ivfhrt")}

    shard_size = shard_size  # e.g. 100_000
    vec_dim = model.get_sentence_embedding_dimension()
    xb = np.zeros((shard_size, vec_dim), dtype="float32")
    metas = []
    shard_count = 0
    write_count = 0

    print("Prepping FAISS")
    # initialize FAISS: IVF-HNSW (approx HNSW within inverted lists)
    nlist = 1024
    quantizer = faiss.IndexFlatL2(vec_dim)
    index = faiss.IndexIVFFlat(quantizer, vec_dim, nlist)
    print(f"Trained: {index.is_trained}")

    xb = np.random.random((shard_size, vec_dim)).astype("float32")
    index.train(xb)

    # TODO: retrain!
    # index.train(np.zeros((1, vec_dim), dtype='float32'))

    vector_id_counter = 0

    print(f"Ingesting {len(md_files)} docs")
    for doc_id, md_path in enumerate(md_files):
        text = read_md(md_path)
        chunks = chunk_text(text, max_tokens=max_tokens)
        embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

        for chunk_id, (chunk, emb) in enumerate(zip(chunks, embs)):
            i = write_count % shard_size
            xb[i] = emb
            metas.append(
                {
                    "vector_id": vector_id_counter,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source": md_path,
                    "text": chunk,
                }
            )
            write_count += 1
            vector_id_counter += 1

            # once we fill a shard: flush to Parquet and add to index
            if write_count % shard_size == 0:
                tbl = pa.Table.from_pylist(
                    [{**m, "embedding": m_emb.tolist()} for m, m_emb in zip(metas, xb)]
                )
                shard_file = str(shards_dir / f"shard_{shard_count:04d}.parquet")
                print(f"Writing shard: {shard_file}")
                pq.write_table(tbl, shard_file)
                manifest["parquet_shards"].append(shard_file)

                # index vectors in FAISS
                index.add(xb)
                shard_count += 1
                metas.clear()
                xb.fill(0)

    # handle final partial shard
    rem = write_count % shard_size
    if rem:
        tbl = pa.Table.from_pylist(
            [{**m, "embedding": xb[i].tolist()} for i, m in enumerate(metas)]
        )
        shard_file = str(shards_dir / f"shard_{shard_count:04d}.parquet")

        print(f"Writing final shard: {shard_file}")
        pq.write_table(tbl, shard_file)
        manifest["parquet_shards"].append(shard_file)
        index.add(xb[:rem])

    print("Saving FAISS index")
    faiss.write_index(index, manifest["faiss_index"])

    print("Saving shards manifest")
    with open(Path(output_dir) / "manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(
        f"Built {write_count} chunks, {len(manifest['parquet_shards'])} shards, index at {manifest['faiss_index']}"
    )


if __name__ == "__main__":
    main()
