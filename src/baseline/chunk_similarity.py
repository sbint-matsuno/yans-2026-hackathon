"""Baseline predictor using embedding similarity against paper chunks.

Like `embedding_similarity.py`, but instead of comparing a citation context to
the cited paper's abstract, it compares against every chunk of the cited
paper (see scripts/extract_chunks.py) and takes the maximum cosine similarity
as the pair's score. Uses cl-nagoya/ruri-v3-70m (70M parameters, within the
100M-parameter regulation).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score


MODEL_NAME = "cl-nagoya/ruri-v3-30m"
# Caps padded-batch size so a few outlier long chunks don't blow up memory
# (and trips a known MPS int32-overflow assertion on very large padded tensors).
MAX_SEQ_LENGTH = 512


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_chunks(chunks_path: str) -> dict[str, list[str]]:
    """Load paper chunks from a chunks.jsonl file.

    Returns a mapping from paper_id to its list of chunk texts.
    """
    chunks: dict[str, list[str]] = defaultdict(list)
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            paper_id = entry.get("paper_id", "")
            text = entry.get("text", "")
            if paper_id and text:
                chunks[paper_id].append(text)
    return dict(chunks)


def _embed_chunks(
    model: SentenceTransformer, chunks: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """Encode every paper's chunks, returning paper_id -> (n_chunks, dim) array."""
    paper_ids = list(chunks.keys())
    all_texts: list[str] = []
    offsets: list[tuple[int, int]] = []
    for pid in paper_ids:
        start = len(all_texts)
        all_texts.extend(chunks[pid])
        offsets.append((start, len(all_texts)))

    embs = model.encode(
        all_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    chunk_embs: dict[str, np.ndarray] = {}
    for pid, (start, end) in zip(paper_ids, offsets):
        chunk_embs[pid] = embs[start:end]
    return chunk_embs


def _compute_similarities(
    model: SentenceTransformer,
    records: list[dict],
    chunk_embs: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray, list[int]]:
    """Compute max cosine similarity between each citation context and the
    cited paper's chunks.

    Returns
    -------
    ids : list[str]
        Record IDs in the same order.
    similarities : np.ndarray
        Max cosine similarity score per record (0.0 if the cited paper has no chunks).
    labels : list[int]
        Gold labels (may be empty if label field is absent).
    """
    contexts = [rec["citation_context"] for rec in records]
    ctx_embs = model.encode(
        contexts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
    )

    ids: list[str] = []
    labels: list[int] = []
    similarities = np.zeros(len(records), dtype=np.float32)

    for i, rec in enumerate(records):
        ids.append(rec["id"])
        if "label" in rec:
            labels.append(int(rec["label"]))

        cited_id = rec["cited_paper_id"]
        cand_embs = chunk_embs.get(cited_id)
        if cand_embs is None or len(cand_embs) == 0:
            similarities[i] = 0.0
            continue
        sims = cand_embs @ ctx_embs[i]
        similarities[i] = float(sims.max())

    return ids, similarities, labels


def _optimise_threshold(
    similarities: np.ndarray,
    labels: list[int],
    steps: int = 200,
) -> float:
    """Find the similarity threshold that maximises F1 on the given labels."""
    best_f1 = -1.0
    best_threshold = 0.5
    lo, hi = float(similarities.min()), float(similarities.max())

    for i in range(steps + 1):
        threshold = lo + (hi - lo) * i / steps
        preds = (similarities >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chunk-max-similarity baseline for citation correctness."
    )
    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Path to chunks.jsonl file (see scripts/extract_chunks.py).",
    )
    parser.add_argument(
        "--dev",
        type=str,
        required=True,
        help="Path to dev JSONL file (used for threshold optimisation).",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write prediction JSONL.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the embedding model (default: cpu; MPS can hit size limits on long batches).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Sentence-transformers model name (default: {MODEL_NAME}).",
    )
    args = parser.parse_args()

    print(f"Loading chunks from {args.chunks} ...")
    chunks = _load_chunks(args.chunks)
    n_chunks = sum(len(v) for v in chunks.values())
    print(f"  Loaded {n_chunks} chunks across {len(chunks)} papers.")

    print(f"Loading model: {args.model} (device={args.device}) ...")
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = MAX_SEQ_LENGTH

    print("Embedding chunks ...")
    chunk_embs = _embed_chunks(model, chunks)

    # --- Dev set: compute similarities and optimise threshold ---
    print("Processing dev set ...")
    dev_records = _read_jsonl(args.dev)
    dev_ids, dev_sims, dev_labels = _compute_similarities(model, dev_records, chunk_embs)
    threshold = _optimise_threshold(dev_sims, dev_labels)
    print(f"  Optimised threshold: {threshold:.4f}")

    dev_preds = (dev_sims >= threshold).astype(int)
    dev_f1 = f1_score(dev_labels, dev_preds, zero_division=0)
    print(f"  Dev F1: {dev_f1:.4f}")

    # --- Test set: predict ---
    print("Processing test set ...")
    test_records = _read_jsonl(args.test)
    test_ids, test_sims, _ = _compute_similarities(model, test_records, chunk_embs)
    test_preds = (test_sims >= threshold).astype(int)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec_id, pred, sim in zip(test_ids, test_preds, test_sims):
            f.write(
                json.dumps(
                    {"id": rec_id, "prediction": int(pred), "similarity": float(sim)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Predictions written to {args.output} ({len(test_preds)} records).")

    # Also dump dev-set similarities/predictions for inspection.
    dev_output_path = output_path.with_name(output_path.stem + "_dev" + output_path.suffix)
    with open(dev_output_path, "w", encoding="utf-8") as f:
        for rec_id, pred, sim, label in zip(dev_ids, dev_preds, dev_sims, dev_labels):
            f.write(
                json.dumps(
                    {
                        "id": rec_id,
                        "prediction": int(pred),
                        "similarity": float(sim),
                        "label": int(label),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Dev similarities written to {dev_output_path} ({len(dev_preds)} records).")


if __name__ == "__main__":
    main()
