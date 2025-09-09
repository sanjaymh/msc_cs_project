
import pandas as pd
import numpy as np
import ast
import re, html

from typing import Optional
try:
    import spacy
    _NLP: Optional["spacy.Language"] = None
except Exception:
    spacy = None
    _NLP = None

# constants:
MD_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE   = re.compile(r"`[^`]+`")
MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")  # keep alt, drop URL
MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")    # keep text, drop URL
URL           = re.compile(r"https?://\S+")
HASH          = re.compile(r"\b[0-9a-f]{7,40}\b")
IMG_TAG       = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
HTML_TAG      = re.compile(r"<[^>]+>")  # any remaining HTML tags
MULTI_WS      = re.compile(r"\s+")

def _to_utc_datetime(df: pd.DataFrame, col: str = "created_at") -> pd.DataFrame:
    if col not in df.columns:
        return df
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

def _drop_dupes(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer dropping by 'id' if present; otherwise by 'number'; otherwise full-row duplicates.
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
    elif "number" in df.columns:
        df = df.drop_duplicates(subset=["number"], keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    return df

def _normalize_empty_strings(df: pd.DataFrame, cols=("title","body","labels","reviewers")) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace({"": np.nan}).astype("object")
            # Also treat obvious placeholders as missing
            df[c] = df[c].replace({"None": np.nan, "null": np.nan, "N/A": np.nan})
    return df

def _split_semicolon(col: pd.Series) -> pd.Series:
    # Convert semicolon-separated strings into a list[str], safe for NaN
    def splitter(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return [i.strip() for i in x if str(i).strip()]
        # assume string
        parts = [p.strip() for p in str(x).split(";")]
        return [p for p in parts if p]
    return col.apply(splitter)

def parse_list_col(col: pd.Series) -> pd.Series:
    """Check that DataFrame column contains Python lists (not stringified)."""
    def _parse(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    val = ast.literal_eval(s)
                    if isinstance(val, list):
                        return [str(v).strip().strip("'\"") for v in val if str(v).strip()]
                except Exception:
                    pass
            # split on common delimiters
            for sep in (";", ","):
                if sep in s:
                    return [t.strip().strip("'\"") for t in s.split(sep) if t.strip()]
            return [s.strip().strip("'\"")]
        return [str(x)]
    return col.apply(_parse)

def _unescape_multi(s: str, rounds: int = 3) -> str:
    """Unescape HTML entities repeatedly (handles double-escaped content)."""
    prev = None
    cur = s
    for _ in range(rounds):
        prev = cur
        cur = html.unescape(cur)
        if cur == prev:
            break
    return cur

def clean_body_text(text: str) -> str:
    if text is None:
        return ""
    t = _unescape_multi(str(text))  # handles &lt;img ...&gt; and friends

    # Remove code blocks/inline
    t = MD_CODE_FENCE.sub(" ", t)
    t = INLINE_CODE.sub(" ", t)

    # Markdown images/links: keep visible text/alt, drop URLs
    t = MARKDOWN_IMAGE.sub(lambda m: f" {m.group(1).strip()} " if m.group(1) else " ", t)
    t = MARKDOWN_LINK.sub(r"\1", t)

    # HTML <img ...> : keep alt="" text if present (else drop)
    def _img_alt_repl(m):
        tag = m.group(0)
        altm = re.search(r'alt\s*=\s*"([^"]*)"|alt\s*=\s*\'([^\']*)\'', tag, re.IGNORECASE | re.DOTALL)
        if altm:
            val = (altm.group(1) or altm.group(2) or "").strip()
            return f" {val} " if val else " "
        return " "
    t = IMG_TAG.sub(" ", t)

    # Strip any other HTML tags (robust across newlines)
    t = HTML_TAG.sub(" ", t)

    # Remove raw URLs and commit hashes
    t = URL.sub(" ", t)
    t = HASH.sub(" ", t)

    # Strip markdown bullets/headers/quotes
    t = re.sub(r"^[#>\-\*\+]\s*", " ", t, flags=re.MULTILINE)

    # Normalize whitespace
    t = MULTI_WS.sub(" ", t).strip()
    return t

def _get_nlp():
    global _NLP
    if _NLP is None and spacy is not None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = None
    return _NLP

def spacy_keyphrases(text: str, max_words: int = 3, max_phrases: int = 5) -> str:
    if not text:
        return ""
    nlp = _get_nlp()
    if nlp is None:
        return ""
    doc = nlp(text)
    chunks = [c.text for c in doc.noun_chunks if len(c.text.split()) <= max_words]
    if chunks:
        return " | ".join(chunks[:max_phrases])
    kws = [t.lemma_ for t in doc if t.pos_ in ("NOUN","PROPN","ADJ")]
    return " ".join(kws[:max_phrases])

def clean_data(
    df: pd.DataFrame,
    remove_self_reviews: bool = False,
    split_lists: bool = True,
    lowercase_tokens: bool = False,
    trim_tokens: bool = True,
) -> pd.DataFrame:
    """Clean PR dataset in a reproducible way.

    Steps:
      1) Parse created_at to UTC datetime.
      2) Drop duplicate PRs (by id, else number, else full row).
      3) Normalize empty strings/placeholder text to NaN.
      4) Optionally split semicolon-separated 'labels' and 'reviewers' into lists.
      5) Optionally lowercase tokens and trim whitespace.
      6) Create `body_clean` (HTML/Markdown/code stripped).
      7) Create `keyphrases` from `body_clean` (spaCy noun-chunks/keywords).
    """
    df = df.copy()

    # 1) Parse created_at
    df = _to_utc_datetime(df)

    # 2) Drop duplicates
    before = len(df)
    df = _drop_dupes(df)
    df.attrs["dropped_duplicates"] = before - len(df)

    # 3) Normalize empties
    df = _normalize_empty_strings(df)

    # 4) Remove self reviews
    if remove_self_reviews and "author" in df.columns and "reviewers" in df.columns:
        df = df[df["author"] != df["reviewers"]]

    # 5) Split lists
    if split_lists:
        if "labels" in df.columns:
            df["labels_list"] = _split_semicolon(df["labels"])
        if "reviewers" in df.columns:
            df["reviewers_list"] = _split_semicolon(df["reviewers"])

        # Token normalization
        if lowercase_tokens or trim_tokens:
            def norm_tokens(lst):
                out = []
                for t in lst:
                    t2 = str(t)
                    if trim_tokens:
                        t2 = t2.strip()
                    if lowercase_tokens:
                        t2 = t2.lower()
                    if t2:
                        out.append(t2)
                seen, uniq = set(), []
                for t in out:
                    if t not in seen:
                        seen.add(t)
                        uniq.append(t)
                return uniq
            if "labels_list" in df.columns:
                df["labels_list"] = df["labels_list"].apply(norm_tokens)
            if "reviewers_list" in df.columns:
                df["reviewers_list"] = df["reviewers_list"].apply(norm_tokens)

    # 6) Body cleaning → body_clean
    if "body" in df.columns:
        df["body_clean"] = df["body"].apply(clean_body_text)
    else:
        df["body_clean"] = ""

    # 7) spaCy keyphrases from body_clean (safe if spaCy missing)
    df["keyphrases"] = df["body_clean"].apply(spacy_keyphrases)

    return df


def quick_health_report(df: pd.DataFrame) -> dict:
    """Return quick metrics to verify cleanliness."""
    report = {}
    report["rows"] = len(df)
    report["cols"] = df.shape[1]
    for key in ("id","number"):
        if key in df.columns:
            report[f"duplicate_{key}s"] = int(df[key].duplicated().sum())
    report["total_nulls"] = int(df.isna().sum().sum())
    if "created_at" in df.columns:
        try:
            report["unparseable_dates"] = int(df["created_at"].isna().sum())
            report["date_min"] = str(df["created_at"].min())
            report["date_max"] = str(df["created_at"].max())
        except Exception:
            report["unparseable_dates"] = "unknown"
    if "labels_list" in df.columns:
        report["avg_labels_per_pr"] = float(df["labels_list"].apply(len).mean())
    if "reviewers_list" in df.columns:
        report["avg_reviewers_per_pr"] = float(df["reviewers_list"].apply(len).mean())
    return report
