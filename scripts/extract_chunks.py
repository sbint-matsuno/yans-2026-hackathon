"""Extract plain text from papers.jsonl's tex_content and split it into chunks.

Strips LaTeX commands/environments from each paper's tex_content. The text
typically mixes Japanese and English (e.g. a Japanese abstract followed by an
English one), so each paragraph is language-detected with py3langid and
grouped into runs of consecutive same-language paragraphs. Each run is then
sentence-segmented with the matching pysbd model (ja/en) and grouped into
chunks of roughly --chunk-size characters, tagged with their language.

Usage:
    python scripts/extract_chunks.py --papers data/papers.jsonl --output data/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import py3langid as langid
import pysbd

# Only Japanese and English are expected in this corpus; restrict langid to
# these to avoid misclassifying short/noisy paragraphs as unrelated languages.
langid.set_languages(["ja", "en"])

# pysbd segmenters are stateless and slow to construct repeatedly.
_SEGMENTERS = {
    "ja": pysbd.Segmenter(language="ja", clean=False),
    "en": pysbd.Segmenter(language="en", clean=False),
}

# Environments whose content is not prose and should be dropped entirely.
DROP_ENVIRONMENTS = [
    "equation", "equation*", "align", "align*", "eqnarray", "eqnarray*",
    "table", "table*", "figure", "figure*", "tabular", "array",
    "verbatim", "lstlisting", "thebibliography",
]

# Commands whose argument is prose and should be kept (unwrapped).
KEEP_COMMAND_TEXT = {
    "title", "author", "section", "subsection", "subsubsection",
    "paragraph", "caption", "footnote", "emph", "textbf", "textit",
    "jabstract", "abstract",
}


def strip_latex(tex: str) -> str:
    """Convert a LaTeX source string into plain prose text."""
    text = tex

    # Remove comments (unescaped % to end of line).
    text = re.sub(r"(?<!\\)%.*", "", text)

    # Drop non-prose environments entirely (including their content).
    for env in DROP_ENVIRONMENTS:
        pattern = re.compile(
            r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}",
            re.DOTALL,
        )
        text = pattern.sub(" ", text)

    # Drop math mode content: $$...$$, $...$, \[...\], \(...\).
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.DOTALL)

    # Unwrap remaining \begin{env}/\end{env} markers, keeping inner content.
    text = re.sub(r"\\begin\{[^}]*\}(\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\end\{[^}]*\}", " ", text)

    # Repeatedly unwrap known prose commands: \cmd{...} -> ...
    for _ in range(5):
        text = re.sub(
            r"\\(?:" + "|".join(re.escape(c) for c in KEEP_COMMAND_TEXT) + r")\*?\s*\{([^{}]*)\}",
            r"\1",
            text,
        )

    # Drop any other command with a braced argument, keeping only the arg text
    # for simple cases, or dropping both for commands that carry no prose
    # (e.g. \affiref{KUEE}, \newcommand{...}{...}, \setcounter{...}{...}).
    text = re.sub(r"\\newcommand\{[^{}]*\}(\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\setcounter\{[^{}]*\}\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z@]+\{[^{}]*\}\{[^{}]*\}", " ", text)  # two-arg commands
    text = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?\{([^{}]*)\}", r"\2", text)  # one-arg: keep content

    # Remove any remaining bare commands (no args), e.g. \headauthor, \\, \item.
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)

    # Remove leftover braces and LaTeX line breaks.
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("\\\\", " ")

    # Collapse whitespace. The source hard-wraps lines at a fixed width, so a
    # single "\n" is usually just mid-paragraph line-wrapping, not a real
    # paragraph break — only a blank line (2+ newlines) marks an actual
    # paragraph boundary. Mark those first, then flatten wrap-only newlines
    # into spaces so sentences aren't cut at the wrap column.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\x00", text)

    def _join_wrapped_line(match: re.Match) -> str:
        # English wraps at word boundaries, so the break needs a space to
        # avoid gluing two words together; Japanese wraps mid-word/mid-phrase
        # with no space in the original, so joining with nothing is correct.
        before, after = match.group(1), match.group(2)
        if before.isascii() and after.isascii():
            return before + " " + after
        return before + after

    text = re.sub(r"(\S?)\n(\S?)", _join_wrapped_line, text)
    text = text.replace("\x00", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

    return text


def detect_language(paragraph: str) -> str:
    """Detect whether a paragraph is Japanese or English."""
    lang, _ = langid.classify(paragraph)
    return lang


def group_by_language(text: str) -> list[tuple[str, str]]:
    """Split text into paragraphs, detect each one's language, and merge
    consecutive same-language paragraphs into (lang, text) runs."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    runs: list[tuple[str, str]] = []
    for para in paragraphs:
        lang = detect_language(para)
        if runs and runs[-1][0] == lang:
            runs[-1] = (lang, runs[-1][1] + "\n" + para)
        else:
            runs.append((lang, para))

    return runs


def chunk_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    """Group sentences into chunks of at least ~chunk_size characters each."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        current.append(sent)
        current_len += len(sent)
        if current_len >= chunk_size:
            chunks.append("".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append("".join(current))

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from tex_content and chunk it with pysbd."
    )
    parser.add_argument("--papers", type=str, required=True, help="Path to papers.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to output chunks JSONL")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Default target minimum characters per chunk (used if a per-language size is not set)",
    )
    parser.add_argument(
        "--chunk-size-ja",
        type=int,
        default=50,
        help="Target minimum characters per chunk for Japanese runs (defaults to --chunk-size)",
    )
    parser.add_argument(
        "--chunk-size-en",
        type=int,
        default=150,
        help="Target minimum characters per chunk for English runs (defaults to --chunk-size)",
    )
    args = parser.parse_args()

    chunk_size_by_lang = {
        "ja": args.chunk_size_ja if args.chunk_size_ja is not None else args.chunk_size,
        "en": args.chunk_size_en if args.chunk_size_en is not None else args.chunk_size,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_papers = 0
    n_chunks = 0
    with open(args.papers, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            paper = json.loads(line)
            paper_id = paper.get("paper_id", "")
            tex = paper.get("tex_content", "") or ""

            text = strip_latex(tex)
            runs = group_by_language(text)

            chunk_id = 0
            for lang, run_text in runs:
                segmenter = _SEGMENTERS[lang]
                sentences = segmenter.segment(run_text)
                chunks = chunk_sentences(sentences, chunk_size_by_lang[lang])

                for chunk in chunks:
                    fout.write(
                        json.dumps(
                            {
                                "paper_id": paper_id,
                                "chunk_id": chunk_id,
                                "lang": lang,
                                "text": chunk,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    chunk_id += 1
                    n_chunks += 1
            n_papers += 1

    print(f"Processed {n_papers} papers -> {n_chunks} chunks written to {output_path}")


if __name__ == "__main__":
    main()
