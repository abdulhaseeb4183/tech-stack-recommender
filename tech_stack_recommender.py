"""
Tech Stack Recommender — Content-Based Filtering via TF-IDF & Cosine Similarity
================================================================================

A production-ready, object-oriented recommendation engine that maps user skills
onto a continuous vector space built from professional job-role profiles.  Rather
than relying on binary/Jaccard tag matching, the engine uses the classic
Input → Process → Output (IPO) framework:

    INPUT   — ingest a CSV corpus of job roles + skills, plus a user skill list
    PROCESS — vectorize both with TF-IDF, then rank via cosine similarity
    OUTPUT  — return the Top-N most relevant job roles with match scores

Author  : Senior ML Engineer
Version : 1.0.0
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────────────────────────
# RECOMMENDER ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class TechStackRecommender:
    """Content-Based Filtering engine that recommends job roles for a given
    skill set by projecting both the corpus and the user profile into a
    shared TF-IDF vector space and ranking with cosine similarity.

    Attributes
    ----------
    dataset_path : Path
        Filesystem path to the CSV file containing job-role profiles.
    dataframe : pd.DataFrame | None
        The loaded and validated dataset (populated after ``load_data``).
    vectorizer : TfidfVectorizer
        Scikit-learn transformer used for feature extraction.
    tfidf_matrix : sparse matrix | None
        TF-IDF feature matrix for the job-role corpus.
    """

    # ── Construction ─────────────────────────────────────────────────────

    def __init__(self, dataset_path: str = "raw_skills.csv") -> None:
        """Initialise the recommender with a path to the skills CSV.

        Parameters
        ----------
        dataset_path : str
            Relative or absolute path to the CSV dataset.  Defaults to
            ``"raw_skills.csv"`` in the current working directory.
        """
        self.dataset_path: Path = Path(dataset_path)
        self.dataframe: Optional[pd.DataFrame] = None
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            # Treat comma-separated skills as individual tokens
            token_pattern=r"[A-Za-z0-9#+./]+",
            lowercase=True,
            # Penalise ubiquitous skills across the corpus (the "IDF" part)
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,          # apply 1 + log(tf) dampening
        )
        self.tfidf_matrix = None

    # ──────────────────────────────────────────────────────────────────────
    # LAYER 1 — DATA INPUT & INGESTION
    # ──────────────────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """Load and validate the job-role skills dataset from CSV.

        The method enforces the presence of at least the ``Job Role`` and
        ``Skills`` columns, drops any rows whose skills field is empty or
        NaN, and stores the cleaned frame on ``self.dataframe``.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with columns ``['Job Role', 'Skills']``.

        Raises
        ------
        FileNotFoundError
            If the dataset file does not exist at the specified path.
        ValueError
            If required columns are missing or the file is completely empty.
        """
        try:
            if not self.dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at '{self.dataset_path.resolve()}'. "
                    "Please ensure 'raw_skills.csv' is in the working directory."
                )

            df: pd.DataFrame = pd.read_csv(self.dataset_path)

            # ── Schema validation ────────────────────────────────────────
            required_cols = {"Job Role", "Skills"}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(
                    f"Dataset is missing required columns: {missing}. "
                    f"Found columns: {list(df.columns)}"
                )

            # ── Drop rows with missing skills ────────────────────────────
            initial_rows: int = len(df)
            df = df.dropna(subset=["Skills"])
            df = df[df["Skills"].str.strip().astype(bool)]
            dropped: int = initial_rows - len(df)

            if df.empty:
                raise ValueError(
                    "Dataset contains no usable rows after cleaning. "
                    "Every row was either empty or had missing skills."
                )

            if dropped > 0:
                print(f"⚠  Dropped {dropped} row(s) with missing/empty skills.")

            self.dataframe = df.reset_index(drop=True)
            print(
                f"✔  Loaded {len(self.dataframe)} job-role profiles "
                f"from '{self.dataset_path.name}'."
            )
            return self.dataframe

        except FileNotFoundError as exc:
            print(f"\n✖  FILE ERROR: {exc}")
            raise
        except pd.errors.EmptyDataError:
            msg = f"The file '{self.dataset_path.name}' is empty or unreadable."
            print(f"\n✖  DATA ERROR: {msg}")
            raise ValueError(msg)
        except ValueError as exc:
            print(f"\n✖  VALIDATION ERROR: {exc}")
            raise

    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def validate_user_skills(user_skills: List[str]) -> List[str]:
        """Validate and sanitise the user-supplied skill list.

        Parameters
        ----------
        user_skills : List[str]
            Raw list of skill strings provided by the user.

        Returns
        -------
        List[str]
            Cleaned skill list (whitespace-trimmed, blanks removed).

        Raises
        ------
        TypeError
            If ``user_skills`` is not a list.
        ValueError
            If fewer than 3 valid skills are supplied after sanitisation.
        """
        if not isinstance(user_skills, list):
            raise TypeError(
                f"Expected a list of skill strings, got {type(user_skills).__name__}."
            )

        cleaned: List[str] = [s.strip() for s in user_skills if isinstance(s, str) and s.strip()]

        if len(cleaned) < 3:
            raise ValueError(
                f"A minimum of 3 skills is required; received {len(cleaned)} "
                f"valid skill(s) after sanitisation: {cleaned}"
            )

        return cleaned

    # ──────────────────────────────────────────────────────────────────────
    # LAYER 2 — CORE PROCESSING (SIMILARITY ENGINE)
    # ──────────────────────────────────────────────────────────────────────

    def build_tfidf_matrix(self) -> None:
        """Fit the TF-IDF vectorizer on the job-role skills corpus.

        After this call ``self.tfidf_matrix`` contains the sparse
        document-term matrix where:

        * **TF (Term Frequency)** — how often a skill appears in a given
          role's profile, dampened with ``1 + log(tf)`` to reduce the
          influence of repeated mentions.
        * **IDF (Inverse Document Frequency)** — ``log(N / df_t) + 1``,
          down-weighting skills that appear across many roles (e.g.
          "Python", "Git") while amplifying rare, discriminative skills
          (e.g. "Solidity", "RTOS").

        Raises
        ------
        RuntimeError
            If called before ``load_data()``.
        """
        if self.dataframe is None:
            raise RuntimeError(
                "No data loaded. Call 'load_data()' before building the TF-IDF matrix."
            )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.dataframe["Skills"]
        )
        vocab_size: int = len(self.vectorizer.vocabulary_)
        print(
            f"✔  TF-IDF matrix built — shape {self.tfidf_matrix.shape} "
            f"({vocab_size} unique terms in vocabulary)."
        )

    # ──────────────────────────────────────────────────────────────────────

    def compute_similarity(
        self, user_skills: List[str]
    ) -> List[Tuple[str, float]]:
        """Project the user profile into the corpus vector space and compute
        cosine similarity against every job-role vector.

        **Why cosine similarity?**  Unlike Euclidean distance, cosine
        similarity measures the *angle* between two vectors, making it
        invariant to vector magnitude.  A job role with 15 skills and a
        user profile with 3 skills are compared purely on directional
        alignment, not on list length.  The output is naturally bounded
        in ``[0.0, 1.0]`` for non-negative TF-IDF vectors.

        **Cold-start handling:**  If the user's skills produce a
        zero-vector (none of the terms exist in the corpus vocabulary),
        the method returns an empty list and prints a warning rather
        than propagating a mathematical error.

        Parameters
        ----------
        user_skills : List[str]
            Validated list of user skills (≥ 3 items).

        Returns
        -------
        List[Tuple[str, float]]
            Pairs of ``(job_role, similarity_score)`` sorted in
            **descending** order of relevance.

        Raises
        ------
        RuntimeError
            If the TF-IDF matrix has not been built yet.
        """
        if self.tfidf_matrix is None:
            raise RuntimeError(
                "TF-IDF matrix not initialised. "
                "Call 'build_tfidf_matrix()' first."
            )

        # ── Construct a pseudo-document from the user's skill list ───────
        user_profile_text: str = ", ".join(user_skills)
        user_vector = self.vectorizer.transform([user_profile_text])

        # ── Cold-start detection ─────────────────────────────────────────
        if user_vector.nnz == 0:
            print(
                "\n⚠  COLD START WARNING\n"
                "   None of your skills matched the corpus vocabulary.\n"
                "   Provided skills: " + str(user_skills) + "\n"
                "   Try using more common technical terms (e.g. 'Python',\n"
                "   'Machine Learning', 'Docker').\n"
            )
            return []

        # ── Cosine similarity: user vector vs. every job-role vector ─────
        similarity_scores = cosine_similarity(
            user_vector, self.tfidf_matrix
        ).flatten()

        # ── Map scores back to job roles and sort descending ─────────────
        scored_roles: List[Tuple[str, float]] = []
        for idx, score in enumerate(similarity_scores):
            role: str = self.dataframe.iloc[idx]["Job Role"]
            scored_roles.append((role, float(score)))

        scored_roles.sort(key=lambda pair: pair[1], reverse=True)
        return scored_roles

    # ──────────────────────────────────────────────────────────────────────
    # LAYER 3 — OUTPUT & RANKING
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def display_recommendations(
        scored_roles: List[Tuple[str, float]],
        top_n: int = 3,
    ) -> None:
        """Pretty-print the top-N recommended job roles.

        Parameters
        ----------
        scored_roles : List[Tuple[str, float]]
            Sorted list of ``(role, score)`` pairs from
            ``compute_similarity()``.
        top_n : int
            Number of recommendations to display (default **3**).
        """
        if not scored_roles:
            print(
                "\n╔══════════════════════════════════════════════════════════╗"
                "\n║  No recommendations available.                         ║"
                "\n║  Please refine your skills or check the dataset.       ║"
                "\n╚══════════════════════════════════════════════════════════╝"
            )
            return

        top_results: List[Tuple[str, float]] = scored_roles[:top_n]

        header = "🎯  TOP 3 RECOMMENDED JOB ROLES FOR YOUR SKILL SET"
        divider = "═" * 58

        print(f"\n╔{divider}╗")
        print(f"║  {header:<56} ║")
        print(f"╠{divider}╣")

        for rank, (role, score) in enumerate(top_results, start=1):
            pct: str = f"{score * 100:.1f}%"
            bar_len: int = int(score * 30)
            bar: str = "█" * bar_len + "░" * (30 - bar_len)

            print(f"║                                                          ║")
            print(f"║   #{rank}  {role:<40}        ║")
            print(f"║       Match Score : {pct:>6}                              ║")
            print(f"║       Confidence  : [{bar}]  ║")

            if rank < len(top_results):
                print(f"║  {'─' * 54}  ║")

        print(f"║                                                          ║")
        print(f"╚{divider}╝")

    # ──────────────────────────────────────────────────────────────────────
    # CONVENIENCE — FULL PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    def recommend(
        self, user_skills: List[str], top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """Execute the full IPO pipeline end-to-end.

        1. **Input**   — validate user skills.
        2. **Process** — load data, build TF-IDF, compute cosine similarity.
        3. **Output**  — display and return the top-N recommendations.

        Parameters
        ----------
        user_skills : List[str]
            The user's current technical skill set (minimum 3).
        top_n : int
            How many recommendations to return (default 3).

        Returns
        -------
        List[Tuple[str, float]]
            The top-N ``(job_role, score)`` pairs.
        """
        # ── INPUT ────────────────────────────────────────────────────────
        cleaned_skills: List[str] = self.validate_user_skills(user_skills)
        print(f"\n🔍  User skills accepted: {cleaned_skills}\n")

        # ── PROCESS ──────────────────────────────────────────────────────
        self.load_data()
        self.build_tfidf_matrix()
        scored_roles: List[Tuple[str, float]] = self.compute_similarity(cleaned_skills)

        # ── OUTPUT ───────────────────────────────────────────────────────
        self.display_recommendations(scored_roles, top_n=top_n)
        return scored_roles[:top_n]


# ──────────────────────────────────────────────────────────────────────────────
# DRIVER / DEMO
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Demonstrate the Tech Stack Recommender with sample user profiles."""

    print("=" * 62)
    print("   TECH STACK RECOMMENDER — Content-Based Filtering Engine")
    print("   Using TF-IDF Vectorization & Cosine Similarity")
    print("=" * 62)

    recommender = TechStackRecommender(dataset_path="raw_skills.csv")

    # ── Demo 1: ML-focused user ──────────────────────────────────────────
    print("\n" + "─" * 62)
    print("  DEMO 1 — ML / AI Profile")
    print("─" * 62)
    recommender.recommend(
        user_skills=["Python", "Machine Learning", "Deep Learning",
                      "TensorFlow", "Natural Language Processing"]
    )

    # ── Demo 2: DevOps-focused user ──────────────────────────────────────
    print("\n" + "─" * 62)
    print("  DEMO 2 — DevOps / Cloud Profile")
    print("─" * 62)
    recommender.recommend(
        user_skills=["Docker", "Kubernetes", "AWS",
                      "Terraform", "CI/CD", "Linux"]
    )

    # ── Demo 3: Frontend-focused user ────────────────────────────────────
    print("\n" + "─" * 62)
    print("  DEMO 3 — Frontend / Web Profile")
    print("─" * 62)
    recommender.recommend(
        user_skills=["JavaScript", "React", "CSS",
                      "TypeScript", "Responsive Design"]
    )

    # ── Demo 4: Cold-start scenario (true zero-vector) ──────────────────
    print("\n" + "─" * 62)
    print("  DEMO 4 — Cold Start Scenario (completely unknown skills)")
    print("─" * 62)
    recommender.recommend(
        user_skills=["Esperanto", "Origami", "Beekeeping"]
    )

    # ── Demo 5: Insufficient skills (error handling) ─────────────────────
    print("\n" + "─" * 62)
    print("  DEMO 5 — Validation Error (fewer than 3 skills)")
    print("─" * 62)
    try:
        recommender.recommend(user_skills=["Python"])
    except ValueError as e:
        print(f"\n✖  Caught expected error: {e}")


if __name__ == "__main__":
    main()
