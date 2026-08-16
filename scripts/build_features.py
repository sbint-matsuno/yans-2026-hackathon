"""Build feature vectors for train/dev/test splits (Task 1).

Combines the naive TF-IDF features from notebooks/task1_baseline.ipynb
(abstract/title char n-gram cosine similarity, numeric-overlap ratio) with an
embedding-based feature: the max cosine similarity between the citation
context and the cited paper's text chunks (see scripts/extract_chunks.py),
computed with cl-nagoya/ruri-v3-30m (<=100M-parameter regulation), the same
pair-of-texts idea used in notebooks/task1_finetune.ipynb (citation_context
vs. paper text), but used here to produce a feature instead of a fine-tuned
classifier.

Usage:
    python scripts/build_features.py \
        --train data/train.jsonl --dev data/dev_labeled.jsonl \
        --test data/dev_leaderboard.jsonl --papers data/papers.jsonl \
        --chunks data/chunks.jsonl --output-dir data/features
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baseline.chunk_similarity import _load_chunks

MODEL_NAME = "cl-nagoya/ruri-v3-30m"
MAX_SEQ_LENGTH = 512
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

FEATURE_NAMES = ["abst類似度", "title類似度", "数値一致率", "chunk類似度"]


def _read_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_papers(papers_path: str) -> dict[str, dict]:
    return {p["paper_id"]: p for p in _read_jsonl(papers_path)}


def _build_tfidf(train_records: list[dict], papers: dict[str, dict]) -> TfidfVectorizer:
    cited_ids = {r["cited_paper_id"] for r in train_records}
    all_texts = [r["citation_context"] for r in train_records] + [
        papers[pid].get("abstract", "") for pid in cited_ids if pid in papers
    ]
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2).fit(all_texts)


def _cos(tfidf: TfidfVectorizer, a: str, b: str) -> float:
    va, vb = tfidf.transform([a]), tfidf.transform([b])
    n = np.sqrt(va.multiply(va).sum()) * np.sqrt(vb.multiply(vb).sum())
    return float(va.multiply(vb).sum() / n) if n > 0 else 0.0


def _tfidf_features(tfidf: TfidfVectorizer, rec: dict, papers: dict[str, dict]) -> list[float]:
    ctx = rec["citation_context"]
    p = papers.get(rec["cited_paper_id"], {})
    abst, title = p.get("abstract", ""), p.get("title", "")
    nums = NUM_RE.findall(ctx)
    num_hit = sum(n in abst for n in nums) / len(nums) if nums else 0.5
    return [_cos(tfidf, ctx, abst), _cos(tfidf, ctx, title), num_hit]


def _chunk_similarity_features(
    model: SentenceTransformer, records: list[dict], chunk_embs: dict[str, np.ndarray]
) -> np.ndarray:
    contexts = [rec["citation_context"] for rec in records]
    ctx_embs = model.encode(
        contexts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )
    sims = np.zeros(len(records), dtype=np.float32)
    for i, rec in enumerate(records):
        cand_embs = chunk_embs.get(rec["cited_paper_id"])
        if cand_embs is None or len(cand_embs) == 0:
            sims[i] = 0.0
            continue
        sims[i] = float((cand_embs @ ctx_embs[i]).max())
    return sims


def _embed_chunks(model: SentenceTransformer, chunks: dict[str, list[str]]) -> dict[str, np.ndarray]:
    paper_ids = list(chunks.keys())
    all_texts: list[str] = []
    offsets: list[tuple[int, int]] = []
    for pid in paper_ids:
        start = len(all_texts)
        all_texts.extend(chunks[pid])
        offsets.append((start, len(all_texts)))

    embs = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32)

    chunk_embs: dict[str, np.ndarray] = {}
    for pid, (start, end) in zip(paper_ids, offsets):
        chunk_embs[pid] = embs[start:end]
    return chunk_embs


def _build_split_features(
    records: list[dict],
    tfidf: TfidfVectorizer,
    papers: dict[str, dict],
    model: SentenceTransformer,
    chunk_embs: dict[str, np.ndarray],
) -> np.ndarray:
    tfidf_feats = np.array([_tfidf_features(tfidf, r, papers) for r in records], dtype=np.float32)
    chunk_sims = _chunk_similarity_features(model, records, chunk_embs)
    return np.concatenate([tfidf_feats, chunk_sims[:, None]], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task 1 feature vectors for train/dev/test.")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--papers", type=str, required=True)
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Loading papers from {args.papers} ...")
    papers = _load_papers(args.papers)

    train_records = _read_jsonl(args.train)
    dev_records = _read_jsonl(args.dev)
    test_records = _read_jsonl(args.test)

    print("Fitting TF-IDF on train ...")
    tfidf = _build_tfidf(train_records, papers)

    print(f"Loading chunks from {args.chunks} ...")
    chunks = _load_chunks(args.chunks)

    print(f"Loading model: {MODEL_NAME} (device={args.device}) ...")
    model = SentenceTransformer(MODEL_NAME, device=args.device)
    model.max_seq_length = MAX_SEQ_LENGTH

    print("Embedding chunks ...")
    chunk_embs = _embed_chunks(model, chunks)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, records in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
        print(f"Building features for {name} ({len(records)} records) ...")
        X = _build_split_features(records, tfidf, papers, model, chunk_embs)
        ids = np.array([r["id"] for r in records])
        save_kwargs = {"X": X, "ids": ids, "feature_names": np.array(FEATURE_NAMES)}
        if all("label" in r for r in records):
            save_kwargs["y"] = np.array([int(r["label"]) for r in records])
        out_path = output_dir / f"{name}.npz"
        np.savez(out_path, **save_kwargs)
        print(f"  Wrote {out_path} (shape={X.shape})")


if __name__ == "__main__":
    main()
