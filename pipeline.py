#!/usr/bin/env python3
"""
Pipeline:
1) Read a text file (paragraphs or any text)
2) Sentence-split ("sentenize") -> <base>_sentenized.txt (one sentence per line)
3) Create overlapping sentence “windows” -> <base>_overlapped.txt
4) Embed each overlapped line into float32 vectors -> <base>_overlapped.emb

Notes:
- Sentence splitting supports English and Dutch via NLTK Punkt.
- Overlaps: for each position i, we emit concatenations of s[i:i+k] for k=1..N.
- Embeddings: uses Sentence-Transformers; default model is multilingual-e5-large-instruct (1024-d).
- The .emb file is a NumPy .npy-formatted array written to a file named *.emb
  (so it keeps shape/dtype metadata, but uses your requested extension).
"""

import argparse
import re
from pathlib import Path

import numpy as np


def sentenize_text(text: str, language: str) -> list[str]:
    # Lazy import so the script still shows a helpful error if nltk isn't installed
    import nltk
    from nltk.tokenize import sent_tokenize

    # Ensure punkt is available
    nltk.download("punkt", quiet=True)

    # Normalize whitespace a bit (keep it simple; tokenizer handles newlines too)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    return [s.strip() for s in sent_tokenize(text, language=language) if s.strip()]


def yield_overlaps(sentences: list[str], max_overlaps: int) -> list[str]:
    """
    Produce overlapped strings:
      for i in [0..len-1]:
        for k in [1..max_overlaps]:
          emit " ".join(sentences[i:i+k]) if in range

    This is the common “overlapping sentences” trick used to improve robustness.
    """
    out = []
    n = len(sentences)
    for i in range(n):
        for k in range(1, max_overlaps + 1):
            j = i + k
            if j <= n:
                out.append(" ".join(sentences[i:j]))
            else:
                break
    return out


def embed_lines(
    lines: list[str],
    model_name: str,
    batch_size: int,
    device: str | None,
    normalize: bool,
    e5_mode: str,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    # E5-family models work best with "query: " / "passage: " prefixes.
    if "e5" in model_name.lower() and e5_mode != "none":
        prefix = "query: " if e5_mode == "query" else "passage: "
        lines = [prefix + s for s in lines]

    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(
        lines,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    return np.asarray(embs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Sentenize -> overlap -> embed. Produces *_sentenized.txt, *_overlapped.txt, *_overlapped.emb"
    )
    parser.add_argument("input", help="Input text file (paragraphs or any text).")

    # Sentence splitting language
    parser.add_argument(
        "--lang",
        choices=["english", "dutch"],
        default="english",
        help="Sentence tokenizer language (default: english).",
    )

    # Overlap control
    parser.add_argument(
        "-n",
        "--num-overlaps",
        type=int,
        default=10,
        help="Max overlap window size in sentences (default: 10).",
    )

    # Embedding options
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-large-instruct",
        help="Sentence embedding model (default: intfloat/multilingual-e5-large-instruct).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64).")
    parser.add_argument("--device", default=None, help='Force device: "cpu", "cuda", "mps" (default: auto).')
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings (recommended for cosine similarity).",
    )
    parser.add_argument(
        "--e5-mode",
        choices=["none", "query", "passage"],
        default="passage",
        help="If using an E5 model, prefix lines with 'query:' or 'passage:' (default: passage).",
    )

    # Output directory override (optional)
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory. Default: alongside input file.",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    outdir = Path(args.outdir) if args.outdir else in_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    base = in_path.stem  # file name without extension
    sent_path = outdir / f"{base}_sentenized.txt"
    ovl_path = outdir / f"{base}_overlapped.txt"
    emb_path = outdir / f"{base}_overlapped.emb"

    # 1) Read + sentenize
    text = in_path.read_text(encoding="utf-8")
    sentences = sentenize_text(text, language=args.lang)
    if not sentences:
        raise ValueError("No sentences produced (input may be empty or whitespace).")

    sent_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    print(f"Wrote: {sent_path}  ({len(sentences)} sentences)")

    # 2) Overlap
    overlapped = yield_overlaps(sentences, max_overlaps=args.num_overlaps)

    # Deduplicate + sort for reproducibility (like your example script)
    overlapped = sorted(set(overlapped))

    ovl_path.write_text("\n".join(overlapped) + "\n", encoding="utf-8")
    print(f"Wrote: {ovl_path}  ({len(overlapped)} overlapped lines; max window={args.num_overlaps})")

    # 3) Embed overlapped lines
    embeddings = embed_lines(
        overlapped,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        normalize=args.normalize,
        e5_mode=args.e5_mode,
    )
    print(f"Embeddings shape: {embeddings.shape}, dtype={embeddings.dtype}")

    # Write .npy-format data into a file named *.emb (keeps shape/dtype metadata)
    with open(emb_path, "wb") as f:
        np.save(f, embeddings)
    print(f"Wrote: {emb_path}")

    # (Optional sanity check)
    # Expect 1024 dims for the default E5-large-instruct model.
    if "e5-large" in args.model.lower() and embeddings.shape[1] != 1024:
        print(f"WARNING: expected 1024 dims for E5-large, got {embeddings.shape[1]}.")


if __name__ == "__main__":
    main()

