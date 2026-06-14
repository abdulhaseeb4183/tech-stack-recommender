"""
Flask Web Application for the Tech Stack Recommender
=====================================================
Serves a professional UI that wraps the TechStackRecommender engine,
exposing REST endpoints for recommendations, dataset browsing, and
corpus vocabulary inspection.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

from tech_stack_recommender import TechStackRecommender

# ── Application factory ─────────────────────────────────────────────────────

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Boot the recommender once at startup ─────────────────────────────────────

recommender = TechStackRecommender(
    dataset_path=os.path.join(os.path.dirname(__file__), "raw_skills.csv")
)
recommender.load_data()
recommender.build_tfidf_matrix()


# ── Extract all unique skills from the corpus (for autocomplete) ─────────────

def _extract_corpus_skills() -> List[str]:
    """Return a sorted list of every unique skill mentioned in the dataset."""
    all_skills: set[str] = set()
    for skills_str in recommender.dataframe["Skills"]:
        for skill in skills_str.split(","):
            cleaned = skill.strip()
            if cleaned:
                all_skills.add(cleaned)
    return sorted(all_skills)


CORPUS_SKILLS: List[str] = _extract_corpus_skills()


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main single-page application."""
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Return Top-N recommendations for the supplied skills.

    Request JSON
    -------------
    { "skills": ["Python", "Docker", "Machine Learning"], "top_n": 3 }

    Response JSON
    --------------
    {
      "success": true,
      "user_skills": [...],
      "recommendations": [
        { "rank": 1, "role": "...", "score": 0.62, "pct": "62.0%" , "skills": "..."},
        ...
      ],
      "total_roles": 20,
      "cold_start": false
    }
    """
    try:
        data: Dict[str, Any] = request.get_json(force=True)
        user_skills: List[str] = data.get("skills", [])
        top_n: int = int(data.get("top_n", 3))

        # ── Validate ─────────────────────────────────────────────────────
        cleaned = recommender.validate_user_skills(user_skills)

        # ── Compute similarity ───────────────────────────────────────────
        scored_roles = recommender.compute_similarity(cleaned)

        # ── Cold start check ─────────────────────────────────────────────
        if not scored_roles:
            return jsonify({
                "success": True,
                "user_skills": cleaned,
                "recommendations": [],
                "total_roles": len(recommender.dataframe),
                "cold_start": True,
                "message": (
                    "None of your skills matched our corpus vocabulary. "
                    "Try using more common technical terms like 'Python', "
                    "'Machine Learning', or 'Docker'."
                ),
            })

        # ── Build response ───────────────────────────────────────────────
        recommendations = []
        for rank, (role, score) in enumerate(scored_roles[:top_n], start=1):
            role_row = recommender.dataframe[
                recommender.dataframe["Job Role"] == role
            ].iloc[0]
            recommendations.append({
                "rank": rank,
                "role": role,
                "score": round(score, 4),
                "pct": f"{score * 100:.1f}%",
                "skills": role_row["Skills"],
            })

        return jsonify({
            "success": True,
            "user_skills": cleaned,
            "recommendations": recommendations,
            "total_roles": len(recommender.dataframe),
            "cold_start": False,
        })

    except (TypeError, ValueError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal error: {exc}"}), 500


@app.route("/api/skills", methods=["GET"])
def api_skills():
    """Return all unique skills in the corpus (for autocomplete)."""
    return jsonify({"skills": CORPUS_SKILLS})


@app.route("/api/roles", methods=["GET"])
def api_roles():
    """Return all job roles with their skills (for the dataset explorer)."""
    roles = []
    for _, row in recommender.dataframe.iterrows():
        roles.append({
            "role": row["Job Role"],
            "skills": [s.strip() for s in row["Skills"].split(",")],
        })
    return jsonify({"roles": roles, "total": len(roles)})


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, port=5000)
