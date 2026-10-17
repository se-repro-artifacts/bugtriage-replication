"""Retrieval over the TRAIN split, used by single-round RAG and by the agent's search tool.

TF-IDF with the same vectoriser family as the classical baseline (1-2 grams, sublinear tf,
200k features), L2-normalised so the inner product is cosine similarity. The train matrix is
small (11021 x 200k sparse) and is kept in memory.
"""
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = os.path.join(os.path.dirname(__file__), "..")
DESC_CHARS = 2000
URL_RE = re.compile(r"https?://\S+")


def text_of(row, mode="sd", desc_chars=DESC_CHARS):
    text = row["summary"]
    if mode == "sd" and row.get("description"):
        text += "\n" + URL_RE.sub(" URL ", row["description"][:desc_chars])
    return text


class Retriever:
    """Indexes the train split; exposes similarity search and per-team recency lookups."""

    def __init__(self, rows, label_key="label_team", text_mode="sd"):
        self.label_key = label_key
        self.text_mode = text_mode
        self.rows = [r for r in rows if r["split"] == "train" and r[label_key] not in ("?", "Other")]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000,
                                   sublinear_tf=True, token_pattern=r"(?u)\b\w[\w\-\.]+\b",
                                   dtype=np.float32)
        self.X = normalize(self.vec.fit_transform([text_of(r, text_mode) for r in self.rows]))
        self.by_team = {}
        for row in self.rows:
            self.by_team.setdefault(row[label_key], []).append(row)
        for team in self.by_team:
            # most recently created first; `created` is an ISO-8601 string, so lexical order works
            self.by_team[team].sort(key=lambda r: r["created"], reverse=True)

    def search(self, query, k=10):
        """Return [(row, cosine_similarity), ...] for the k nearest train bugs."""
        if not query.strip():
            return []
        q = normalize(self.vec.transform([query]))
        sims = np.asarray((self.X @ q.T).todense()).ravel()
        order = np.argsort(-sims)[:k]
        return [(self.rows[i], float(sims[i])) for i in order]

    def examples_for(self, text, k=10):
        """The k retrieved train bugs used as in-context examples by the RAG condition."""
        return [row for row, _ in self.search(text, k)]

    def recent_for_team(self, team, n=5):
        return self.by_team.get(team, [])[:n]

    def team_names(self):
        return sorted(self.by_team)

    def component_lines(self, needle, limit=40):
        """Component/team description lines whose team or component name matches `needle`."""
        needle = (needle or "").lower().strip()
        out = []
        for row in self.rows:
            fine = row["label_fine"]
            if needle and needle not in fine.lower() and needle not in str(row[self.label_key]).lower():
                continue
            out.append(f"{fine} -> team {row[self.label_key]}")
            if len(out) >= limit:
                break
        return out
