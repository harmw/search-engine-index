import base64
import json
import time
import faiss
import duckdb
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
BASE = Path(__file__).parent / "index"

class SearchResult(BaseModel):
    doc_id: int
    chunk_id: int
    url: str
    text: str
    score: float
    time_ms: float

@app.on_event("startup")
def startup():
    global manifest, index, model, duck_con

    print("Loading manifest")
    # Load manifest & FAISS
    manifest = json.loads((BASE / "manifest.json").read_text())

    print("Loading FAISS")
    index = faiss.read_index(str(manifest["faiss_index"]))

    # Load embedder
    print("Loading embedder")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize DuckDB with S3/httpfs support if needed
    duck_con = duckdb.connect()
    # If your files are remote, enable HTTPFS and configure credentials:
    # duck_con.execute("INSTALL httpfs; LOAD httpfs;")
    # duck_con.execute("SET s3_region='…'; SET s3_access_key_id='…'; …")

    print("Registering shards")
    # Register the shards as a single table via glob
    shard_glob = str(BASE / "shards" / "*.parquet")
    duck_con.execute(f"""
      CREATE VIEW shards AS
        SELECT vector_id, doc_id, chunk_id, source, text
        FROM read_parquet('{shard_glob}');
    """)

@app.get("/search", response_model=list[SearchResult])
def search(q: str = Query(...), k: int = Query(5, ge=1, le=100)):
    print("Performing search")
    start_time = time.perf_counter()

    q_vec = model.encode([q], convert_to_numpy=True)
    distances, indices = index.search(q_vec, k)
    vec_ids = [int(i) for i in indices[0] if i >= 0]

    if not vec_ids:
        raise HTTPException(404, "No results")

    sql = f"""
      SELECT vector_id, doc_id, chunk_id, source, text
      FROM shards
      WHERE vector_id IN ({','.join(map(str, vec_ids))})
    """
    df = duck_con.execute(sql).fetch_df()

    end_time = time.perf_counter()

    search_time = (end_time - start_time) * 1000
    search_time = f"{search_time:.2f}"

    id_to_score = dict(zip(vec_ids, distances[0]))
    results = []
    for vid in vec_ids:
        row = df[df["vector_id"] == vid].iloc[0]

        source = row.source
        url_part = str(source.split("/")[-1:]).replace(".md", "")
        url = base64.urlsafe_b64decode(url_part)

        results.append(SearchResult(
            doc_id   = int(row.doc_id),
            chunk_id = int(row.chunk_id),
            #source   = row.source,
            url      = url,
            text     = row.text[:100],
            score    = float(id_to_score[vid]),
            time_ms  = search_time,
        ))

    return results

if __name__ == "__main__":
    uvicorn.run("app_with_duckdb:app", host="0.0.0.0", port=8000, reload=True)

