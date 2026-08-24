"""
Dedicated integration test: starts the same FastAPI app used in production
(in-process, via TestClient -- no separate server needed), sends one
"test user" profile to POST /recommendations, and writes the ranked results
to sample_results.csv at the repo root.

The test user's details are defined here AND documented in README.md, per
the assessment's requirement. Replace TEST_USER_PROFILE below with your own
real profile before submitting -- this is currently a labeled placeholder.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.logging import configure_logging
from app.main import app

logger = logging.getLogger(__name__)

# NOTE: placeholder profile -- swap for your real details before submitting,
# and update the corresponding section in README.md to match.
TEST_USER_PROFILE = {
    "user_id": 900001,
    "name": "Test User (placeholder -- replace before submission)",
    "gender": "Male",
    "dob": "1998-03-12",
    "interests": ["Technology", "Music", "Gaming", "Travel"],
    "city": "Gurugram",
    "country": "India",
}

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "sample_results.csv"


def main() -> None:
    configure_logging()

    with TestClient(app) as client:
        response = client.post("/recommendations", json=TEST_USER_PROFILE)
        response.raise_for_status()
        payload = response.json()

    recommendations = payload["recommendations"]
    logger.info(
        "Got %d recommendations for test user %s from a pool of %d candidates",
        len(recommendations), TEST_USER_PROFILE["user_id"], payload["candidate_pool_size"],
    )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "user_id", "name", "city", "country", "score"])
        writer.writeheader()
        for rank, item in enumerate(recommendations, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "user_id": item["user_id"],
                    "name": item["name"],
                    "city": item["city"],
                    "country": item["country"],
                    "score": item["score"],
                }
            )

    logger.info("Wrote results to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
