from __future__ import annotations

import secrets

from lab.core.models import HumanReview
from lab.core.workspace import WorkspaceManager


class ReviewStore:
    def __init__(self, manager: WorkspaceManager):
        self.manager = manager

    def save(self, workspace_id: str, review: HumanReview) -> None:
        path = self.manager.path(workspace_id) / "reviews" / f"review_{secrets.token_hex(8)}.json"
        self.manager.atomic_write_json(path, review.model_dump(mode="json"))

    def list(self, workspace_id: str) -> list[HumanReview]:
        reviews: list[HumanReview] = []
        for path in (self.manager.path(workspace_id) / "reviews").glob("review_*.json"):
            try:
                reviews.append(HumanReview.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(reviews, key=lambda item: item.created_at)
