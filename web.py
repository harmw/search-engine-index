import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import duckdb
import faiss
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
BASE = Path(__file__).parent / "index"


class SearchResult(BaseModel):
    doc_id: int
    chunk_id: int
    url: str
    text: str
    score: float

class APIResponse(BaseModel):
    results: List[SearchResult]
    search_time_ms: str


@asynccontextmanager
async def startup(app: FastAPI):
    global manifest, index, model, duck_con

    logging.info("Loading manifest")
    manifest = json.loads((BASE / "manifest.json").read_text())

    logging.info("Loading FAISS")
    index = faiss.read_index(str(manifest["faiss_index"]))

    logging.info("Loading embedder")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    duck_con = duckdb.connect()
    # duck_con.execute("INSTALL httpfs; LOAD httpfs;")
    # duck_con.execute("SET s3_region='...'; SET s3_access_key_id='...'; ...")

    # Load each shard to read the footer, so we know its schema and row stats
    logging.info("Registering available shards")
    shard_glob = str(BASE / "shards" / "*.parquet")
    duck_con.execute(f"""
      CREATE VIEW shards AS
        SELECT vector_id, doc_id, chunk_id, source, text
        FROM read_parquet('{shard_glob}');
    """)
    yield


app = FastAPI(lifespan=startup)


@app.get("/v0/search", response_model=APIResponse)
def search(q: str = Query(...), k: int = Query(10, ge=1, le=100)):
    logging.info(f"Performing search for {q}")
    start_time = time.perf_counter()

    logging.debug("Embedding query")
    q_vec = model.encode([q], convert_to_numpy=True)

    logging.debug(f"Searching index for k={k}")
    distances, indices = index.search(q_vec, k)
    vec_ids = [int(i) for i in indices[0] if i >= 0]

    if not vec_ids:
        raise HTTPException(404, "No results")

    logging.debug("Searching shards for documents")
    sql = f"""
      SELECT vector_id, doc_id, chunk_id, source, text
      FROM shards
      WHERE vector_id IN ({",".join(map(str, vec_ids))})
    """
    df = duck_con.execute(sql).fetch_df()

    end_time = time.perf_counter()

    search_time = (end_time - start_time) * 1000
    search_time = f"{search_time:.2f}"

    logging.info(f"Search finished in {search_time}ms")

    id_to_score = dict(zip(vec_ids, distances[0]))
    results = []
    for vid in vec_ids:
        row = df[df["vector_id"] == vid].iloc[0]

        source = row.source
        url_part = str(source.split("/")[-1:]).replace(".md", "")
        url = base64.urlsafe_b64decode(url_part)

        results.append(
            SearchResult(
                doc_id=int(row.doc_id),
                chunk_id=int(row.chunk_id),
                # source   = row.source,
                url=url,
                text=row.text[:100],
                score=float(id_to_score[vid]),
            )
        )

    return APIResponse(results=results, search_time_ms=search_time)
