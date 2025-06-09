# Search-engine core

The following diagram/sketch outlines the thought process, desired outcomes, lists constraints and draws a proposed architecture.

![search-engine](./search-engine.png)


## Indexing

The indexer reads _markdown_ content, chunks documents, computes embeddings and stores data in parquet files.

Embeddings are created using the `all-MiniLM-L6-v2` SentenceTransformer model.

We achieve PB index size scalability by sharding parquet files and using FAISS for fast shard and vector lookups.

We use `DuckDB` to access the required shards and use plain SQL to query relevant documents (web pages).

Once the required shard is identified, HTTP Range Requests are used (through DuckDB) to access the required data.

### HNSW

About the _Hierarchical Navigable Small Worlds (NHSW)_ index (FAISS):

> Small-world graph: each vector is a node; nodes are connected to a small number (M) of “neighbor” nodes, forming a proximity graph.
>
> Hierarchy of layers:
>
> Layer 0 contains all nodes.
> 
> Higher layers (1, 2, …) are progressively sparser subsets, chosen randomly.
> 
> Each node appears in all layers up to a randomly assigned maximum level.
> 
> Search procedure (for a query vector q):
> 
> Start at the top layer, with an entry point (often the first node inserted at that layer).
> 
> Greedy search: at each layer, walk the graph by moving to the neighbor closest to q, until no closer neighbor is found.
> 
> Drop down one layer and repeat, using the found node as the new entry point.
> 
> On layer 0, you do a best-first search with a candidate queue of size efSearch to explore a small frontier around that entry point and collect the top-K neighbors.
> 
> Because the graph has “short-cuts” linking distant regions, you find high-quality neighbors in O(log N) hops instead of scanning everything.

### Discussion

n/a


## Query API

Since we use an HNSW we get to do distance calculations.

Inside `DuckDB` we can now compute distances between the query vector and the _semantic answer space_ in our index shards,
returning documents that are semantically close.

### Discussion

Currently, the query setup is slow.

Rust would be better here, being compiled and all.

# Development

This `python` project is using `uv`.

Ingesting scraped data and building the index:

```bash
python3 build-index.py --input ../scraper/scraped_data --out-dir index
```

Result:

```
Loading...
Prepping FAISS
Trained: False
Ingesting 1221 docs
Writing shard: index/shards/shard_0000.parquet
Saving FAISS index
Saving shards manifest
Built 4521 chunks, 1 shards, index at index/faiss/index.ivfhrt
```

To launch the `uvicorn` web api:

```bash
PYTHONPATH=./src uvicorn web:app --reload
```

And to run a query:

```bash
curl 'localhost:8000/v0/search?q=how+does+rust+print+to+console' | jq
```

Result:

```
[
  {
    "doc_id": 2,
    "chunk_id": 1,
    "url": "https://doc.rust-lang.org",
    "text": "[The Rust Style Guide](style-guide/index.html) describes the standard formatting\nof Rust code. Most ",
    "score": 0.8521055579185486,
    "time_ms": 65.93
  },
  {
    "doc_id": 2,
    "chunk_id": 0,
    "url": "https://doc.rust-lang.org",
    "text": "---\ntitle: Rust Documentation\n---\nWelcome to an overview of the documentation provided by the [Rust\n",
    "score": 0.8738031387329102,
    "time_ms": 65.93
  },
  {
    "doc_id": 156,
    "chunk_id": 0,
    "url": "https://docs.rs/x509-tsp/latest/x509_tsp/all.html",
    "text": "---\ntitle: List of all items\ndate: 2021-12-05\n---\nDocs.rs\nx509-tsp-0.1.0\nx509-tsp 0.1.0\nPermalink\nDo",
    "score": 0.936631977558136,
    "time_ms": 65.93
  },
  [..]
```

## Notes on `ruff`

Sorting imports and formatting code style:

```bash
ruff check --select I --fix && ruff format
```

# Additional Resources

Chunking: https://arxiv.org/html/2410.13070v1#abstract

FAISS: https://github.com/facebookresearch/faiss/wiki
