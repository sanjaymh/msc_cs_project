import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack


# -------------------------
# Small helpers
# -------------------------

def _normalize_counter(cnt: Counter) -> Dict[str, float]:
    total = sum(cnt.values()) or 1
    return {k: v / total for k, v in cnt.items()}


def _as_list(x) -> List[str]:
    """Robustly coerce a cell value to a Python list[str]. Handles NaN and numpy arrays.
    Avoids using `or []` which breaks on numpy arrays with ambiguous truth value.
    """
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return [str(x)] if x != "" else []


# -------------------------
# Heuristic Priors
# -------------------------

def build_priors(df_tr: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Build P(reviewer | author) and P(reviewer | label) from TRAIN dataframe.
    Requires columns: 'author', 'labels_list' (List[str]), 'reviewers_list' (List[str]).
    """
    pa = defaultdict(Counter)  # author -> Counter(reviewer)
    pl = defaultdict(Counter)  # label  -> Counter(reviewer)

    for _, row in df_tr.iterrows():
        author = row.get('author', None)
        labels = _as_list(row.get('labels_list', []))
        revs = _as_list(row.get('reviewers_list', []))
        if author is None:
            continue
        for r in revs:
            pa[str(author)][str(r)] += 1
        for lb in labels:
            for r in revs:
                pl[str(lb)][str(r)] += 1

    p_rev_given_author = {a: _normalize_counter(c) for a, c in pa.items()}
    p_rev_given_label  = {l: _normalize_counter(c) for l, c in pl.items()}
    return p_rev_given_author, p_rev_given_label


def build_global_prior(df_tr: pd.DataFrame) -> Dict[str, float]:
    """Global reviewer frequency prior from TRAIN dataframe."""
    cnt = Counter()
    for _, row in df_tr.iterrows():
        for r in _as_list(row.get('reviewers_list', [])):
            cnt[str(r)] += 1
    return _normalize_counter(cnt)


def _approx_global_prior_from_priors(pA: Dict[str, Dict[str, float]],
                                     pL: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Fallback global prior if TRAIN df is not available: average reviewer mass from priors."""
    cnt = Counter()
    for _, dist in pA.items():
        for r, p in dist.items():
            cnt[r] += p
    for _, dist in pL.items():
        for r, p in dist.items():
            cnt[r] += p
    return _normalize_counter(cnt)


def recommend_with_backoff(author: str,
                           labels: List[str],
                           p_rev_given_author: Dict[str, Dict[str, float]],
                           p_rev_given_label: Dict[str, Dict[str, float]],
                           p_global: Dict[str, float],
                           k: int = 3,
                           w_author: float = 0.5,
                           w_labels: float = 0.5) -> List[str]:
    """Combine author and label priors; if empty, back off to global prior."""
    scores = Counter()

    if author in p_rev_given_author:
        for r, p in p_rev_given_author[author].items():
            scores[r] += w_author * p

    labels = _as_list(labels)
    L = max(1, len(labels))
    for lb in labels:
        if lb in p_rev_given_label:
            for r, p in p_rev_given_label[lb].items():
                scores[r] += (w_labels / L) * p

    if not scores:
        for r, p in p_global.items():
            scores[r] += p

    return [r for r, _ in scores.most_common(k)]


# Compatibility wrapper for legacy code

def recommend_reviewers(author: str,
                        labels: List[str],
                        p_rev_given_author: Dict[str, Dict[str, float]],
                        p_rev_given_label: Dict[str, Dict[str, float]],
                        k: int = 3,
                        p_global: Optional[Dict[str, float]] = None) -> List[str]:
    """Legacy signature. Uses backoff automatically."""
    if p_global is None:
        p_global = _approx_global_prior_from_priors(p_rev_given_author, p_rev_given_label)
    return recommend_with_backoff(author, labels, p_rev_given_author, p_rev_given_label, p_global, k=k)


def hit_at_k(test_df: pd.DataFrame,
             p_rev_given_author: Dict[str, Dict[str, float]],
             p_rev_given_label: Dict[str, Dict[str, float]],
             k: int = 3,
             p_global: Optional[Dict[str, float]] = None,
             w_author: float = 0.5,
             w_labels: float = 0.5) -> float:
    """Hit@K with backoff to global prior when author/label signals are missing."""
    if p_global is None:
        p_global = _approx_global_prior_from_priors(p_rev_given_author, p_rev_given_label)

    hits, total = 0, 0
    for _, row in test_df.iterrows():
        truth = set(_as_list(row.get('reviewers_list', [])))
        preds = recommend_with_backoff(
            row.get('author', None),
            _as_list(row.get('labels_list', [])),
            p_rev_given_author,
            p_rev_given_label,
            p_global,
            k=k,
            w_author=w_author,
            w_labels=w_labels,
        )
        if truth and any(p in truth for p in preds):
            hits += 1
        total += 1
    return hits / total if total else 0.0


# -------------------------
# Feature Prep + ML Baseline
# -------------------------

def prepare_features(df: pd.DataFrame):
    """Prepare one-hot author + labels as X, and multi-hot reviewers as Y."""
    X_author = pd.get_dummies(df['author'].fillna('unknown_author'), prefix='author')

    mlb_labels = MultiLabelBinarizer()
    L = mlb_labels.fit_transform(df['labels_list'].apply(_as_list))
    X_labels = pd.DataFrame(L, columns=[f'label::{c}' for c in mlb_labels.classes_], index=df.index)

    X = pd.concat([X_author, X_labels], axis=1)

    mlb_reviewers = MultiLabelBinarizer()
    Y_arr = mlb_reviewers.fit_transform(df['reviewers_list'].apply(_as_list))
    Y = pd.DataFrame(Y_arr, columns=list(mlb_reviewers.classes_), index=df.index)

    return X, Y, mlb_labels, mlb_reviewers

# -------------------------
# extended feature builder (author + labels + optional TF-IDF(body/keyphrases))
# -------------------------
class FeatureBuilder:
    def __init__(self,
                 use_body: bool = True,
                 use_keyphrases: bool = True,
                 body_col: str = "body_clean",
                 keyphr_col: str = "keyphrases",
                 tfidf_params_body: Optional[dict] = None,
                 tfidf_params_keyphr: Optional[dict] = None):
        self.use_body = use_body
        self.use_keyphrases = use_keyphrases
        self.body_col = body_col
        self.keyphr_col = keyphr_col
        self.tfidf_params_body = tfidf_params_body or {"ngram_range": (1,2), "min_df": 3, "max_features": 200_000, "lowercase": True}
        self.tfidf_params_keyphr = tfidf_params_keyphr or {"ngram_range": (1,2), "min_df": 2, "max_features": 50_000,  "lowercase": True}
        self.author_enc: Optional[OneHotEncoder] = None
        self.labels_mlb: Optional[MultiLabelBinarizer] = None
        self.reviewers_mlb: Optional[MultiLabelBinarizer] = None
        self.tfidf_body: Optional[TfidfVectorizer] = None
        self.tfidf_keyphr: Optional[TfidfVectorizer] = None
        self.output_labels_: Optional[List[str]] = None
        self.feature_names_in_: Optional[np.ndarray] = None

    @staticmethod
    def _as_list(x) -> List[str]:
        if x is None: return []
        if isinstance(x, float) and np.isnan(x): return []
        if isinstance(x, (list, tuple, set)): return list(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return [str(x)] if x != "" else []

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        # Author
        self.author_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.author_enc.fit(df["author"].fillna("unknown_author").astype(str).to_numpy().reshape(-1,1))
        # Labels
        self.labels_mlb = MultiLabelBinarizer()
        self.labels_mlb.fit(df["labels_list"].apply(self._as_list))
        # Targets
        self.reviewers_mlb = MultiLabelBinarizer()
        self.reviewers_mlb.fit(df["reviewers_list"].apply(self._as_list))
        self.output_labels_ = list(self.reviewers_mlb.classes_)
        # TF-IDF: body
        if self.use_body:
            self.tfidf_body = TfidfVectorizer(**self.tfidf_params_body)
            self.tfidf_body.fit(df.get(self.body_col, pd.Series([""]*len(df))).fillna("").astype(str))
        # TF-IDF: keyphrases
        if self.use_keyphrases:
            self.tfidf_keyphr = TfidfVectorizer(**self.tfidf_params_keyphr)
            self.tfidf_keyphr.fit(df.get(self.keyphr_col, pd.Series([""]*len(df))).fillna("").astype(str))
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        Xa = self.author_enc.transform(df["author"].fillna("unknown_author").astype(str).to_numpy().reshape(-1,1))
        L = self.labels_mlb.transform(df["labels_list"].apply(self._as_list))
        Xl = csr_matrix(L)
        parts = [Xa, Xl]
        if self.tfidf_body is not None:
            Xb = self.tfidf_body.transform(df.get(self.body_col, pd.Series([""]*len(df))).fillna("").astype(str))
            parts.append(Xb)
        if self.tfidf_keyphr is not None:
            Xk = self.tfidf_keyphr.transform(df.get(self.keyphr_col, pd.Series([""]*len(df))).fillna("").astype(str))
            parts.append(Xk)
        X = hstack(parts, format="csr")
        Y_arr = self.reviewers_mlb.transform(df["reviewers_list"].apply(self._as_list))
        Y = pd.DataFrame(Y_arr, columns=list(self.reviewers_mlb.classes_), index=df.index)
        return X, Y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        self.fit(df)
        return self.transform(df)

def train_logistic_baseline(X: pd.DataFrame, Y: pd.DataFrame):
    """OVR Logistic Regression baseline with degenerate targets dropped."""
    # Keep a copy of label names BEFORE converting to ndarray
    label_names = list(Y.columns) if hasattr(Y, "columns") else []

    # Drop degenerate targets
    if hasattr(Y, "columns"):
        keep = [c for c in Y.columns if (Y[c].sum() > 0) and (Y[c].sum() < len(Y))]
        Y = Y[keep] if len(keep) else Y.iloc[:, :0]
        label_names = list(Y.columns)

    # Ensure dense numeric ndarray for y
    Y_arr = Y.values if hasattr(Y, "values") else np.asarray(Y)
    Y_arr = np.asarray(Y_arr, dtype=float)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(X, Y_arr)

    # Attach metadata if available
    try:
        clf.feature_names_in_ = getattr(clf, "feature_names_in_", None)
    except Exception:
        pass
    clf.output_labels_ = label_names
    return clf

def evaluate_with_threshold(clf, X, Y, thr: float = 0.2) -> dict:
    """Evaluate micro P/R/F1 at a given probability threshold.
       Works with sparse X (csr_matrix) and Y as DataFrame/ndarray.
    """
    # --- Target columns in model-trained order ---
    target_cols = list(getattr(clf, "output_labels_", []))
    if not target_cols:
        return {"micro_f1": 0.0, "micro_precision": 0.0, "micro_recall": 0.0}

    # --- Align Y to target columns (no dependency on X being DataFrame) ---
    if hasattr(Y, "reindex"):
        Y = Y.reindex(columns=target_cols, fill_value=0)
        Y_true = Y.values
    else:
        Y_true = np.asarray(Y)
        if Y_true.ndim == 1:
            Y_true = Y_true.reshape(-1, 1)

    n_samples = X.shape[0]
    n_classes = len(target_cols)
    P = np.zeros((n_samples, n_classes), dtype=float)

    # --- Build probs matrix explicitly (handles OVR estimators) ---
    estimators = getattr(clf, "estimators_", None)
    if estimators is not None and len(estimators) == n_classes:
        for j, est in enumerate(estimators):
            if hasattr(est, "predict_proba"):
                pj = est.predict_proba(X)
                pj = pj[:, 1] if pj.ndim == 2 else pj
            elif hasattr(est, "decision_function"):
                z = est.decision_function(X)
                pj = 1 / (1 + np.exp(-z))
            else:
                pj = est.predict(X)
            P[:, j] = np.asarray(pj).reshape(-1)
    else:
        if hasattr(clf, "predict_proba"):
            probs_list = clf.predict_proba(X)
            cols = []
            for p in probs_list:
                pj = p[:, 1] if (hasattr(p, "ndim") and p.ndim == 2) else p
                cols.append(np.asarray(pj).reshape(-1))
            if cols:
                P = np.column_stack(cols)

    Y_pred = (P >= thr).astype(int)

    from sklearn.metrics import f1_score, precision_score, recall_score
    return {
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "micro_precision": precision_score(Y_true, Y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(Y_true, Y_pred, average="micro", zero_division=0),
    }

def evaluate_multilabel(clf, X, Y, threshold: float = 0.2) -> dict:
    # Target columns in model-trained order
    target_cols = list(getattr(clf, "output_labels_", []))
    if not target_cols:
        return {"micro_f1": 0.0, "micro_precision": 0.0, "micro_recall": 0.0}

    # Align Y to target columns (order), without relying on X.index
    if hasattr(Y, "reindex"):
        Y = Y.reindex(columns=target_cols, fill_value=0)
    else:
        Y = pd.DataFrame(Y, columns=target_cols).fillna(0)

    # Build probabilities matrix P (n_samples, n_classes)
    n_samples = (X.shape[0] if hasattr(X, "shape") else len(Y))
    n_classes = len(target_cols)
    P = np.zeros((n_samples, n_classes), dtype=float)

    estimators = getattr(clf, "estimators_", None)
    if estimators is not None and len(estimators) == n_classes:
        for j, est in enumerate(estimators):
            if hasattr(est, "predict_proba"):
                pj = est.predict_proba(X)
                pj = pj[:, 1] if pj.ndim == 2 else pj
            elif hasattr(est, "decision_function"):
                z = est.decision_function(X)
                pj = 1 / (1 + np.exp(-z))
            else:
                pj = est.predict(X)
            P[:, j] = np.asarray(pj).reshape(-1)
    else:
        # Fallback: OneVsRest predict_proba returns list; stitch columns
        if hasattr(clf, "predict_proba"):
            probs_list = clf.predict_proba(X)
            cols = []
            for p in probs_list:
                pj = p[:, 1] if (hasattr(p, "ndim") and p.ndim == 2) else p
                cols.append(np.asarray(pj).reshape(-1))
            if cols:
                P = np.column_stack(cols)

    Y_true = Y.values if hasattr(Y, "values") else np.asarray(Y)
    Y_pred = (P >= threshold).astype(int)

    return {
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "micro_precision": precision_score(Y_true, Y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(Y_true, Y_pred, average="micro", zero_division=0),
    }

