from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from src.embedder import Embedder  # noqa: E402

EVENT_WEIGHTS = {
    "view": 1.0,
    "wishlist": 2.0,
    "cart": 3.0,
    "purchase": 5.0,
}

@dataclass
class CandidateFrame:
    features: pd.DataFrame
    group_sizes: list[int]


class HybridRecommender:
    def __init__(
        self,
        project_root: Path | None = None,
        candidate_top_k: int = 35,
        blend_weights: dict[str, float] | None = None,
        similarity_threshold: float = 0.18,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.data_dir = self.project_root / "data"
        self.candidate_top_k = candidate_top_k
        self.similarity_threshold = similarity_threshold
        self.blend_weights = blend_weights or {
            "cf_score": 0.34,
            "embedding_similarity": 0.26,
            "price_sensitivity_match": 0.12,
            "style_affinity_score": 0.10,
            "material_preference_score": 0.08,
            "category_match_score": 0.06,
            "historical_ctr": 0.04,
        }
        self.embedder = Embedder(
            data_path=str(self.data_dir / "products_clean.parquet"),
            output_dir=str(self.data_dir),
        )
        self.products = pd.read_parquet(self.data_dir / "products_clean.parquet")
        self.user_features = pd.read_parquet(self.data_dir / "user_features.parquet")
        self.behavior = pd.read_parquet(self.data_dir / "user_behavior_clean.parquet").copy()
        self.behavior["timestamp"] = pd.to_datetime(self.behavior["timestamp"])
        self.embeddings = np.load(self.data_dir / "product_embeddings.npy").astype(np.float32)

        self.products = self.products.sort_values("product_id").reset_index(drop=True)
        self.user_features = self.user_features.copy()
        self.user_features["timestamp"] = pd.to_datetime(self.user_features["timestamp"])
        self.behavior["event_weight"] = self.behavior["event_type"].map(EVENT_WEIGHTS).astype(
            np.float32
        )

        self.product_index = {
            int(product_id): idx for idx, product_id in enumerate(self.products["product_id"])
        }
        self.item_ids = self.products["product_id"].astype(int).tolist()
        self.user_ids = sorted(self.behavior["user_id"].astype(int).unique().tolist())
        self.user_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.user_feature_map = (
            self.user_features.sort_values("user_id").set_index("user_id").to_dict("index")
        )

        self.cf_scores_by_user: dict[int, np.ndarray] = {}
        self.cf_scores_raw_by_user: dict[int, np.ndarray] = {}
        self.user_seen_items: dict[int, set[int]] = {}
        self.user_profile_embeddings: dict[int, np.ndarray] = {}
        self.user_pref_scores: dict[int, dict[str, dict[str, float]]] = {}
        self.user_avg_price: dict[int, float] = {}
        self.item_ctr: dict[int, float] = {}
        self.item_popularity: list[int] = []

        self._fit_from_behavior(self.behavior)

    def _fit_from_behavior(self, interactions: pd.DataFrame) -> None:
        weighted_matrix = np.zeros(
            (len(self.user_ids), len(self.item_ids)),
            dtype=np.float32,
        )
        collapsed = (
            interactions.groupby(["user_id", "product_id"], as_index=False)["event_weight"].sum()
        )
        for row in collapsed.itertuples(index=False):
            user_idx = self.user_index[int(row.user_id)]
            item_idx = self.product_index[int(row.product_id)]
            weighted_matrix[user_idx, item_idx] = float(row.event_weight)

        item_norms = np.linalg.norm(weighted_matrix, axis=0, keepdims=True)
        item_norms[item_norms == 0] = 1.0
        normalized_items = weighted_matrix / item_norms
        item_similarity = normalized_items.T @ normalized_items
        np.fill_diagonal(item_similarity, 0.0)

        for user_id, user_idx in self.user_index.items():
            user_vector = weighted_matrix[user_idx]
            raw_cf_scores = (user_vector @ item_similarity).astype(np.float32)
            seen_items = set(
                interactions.loc[interactions["user_id"] == user_id, "product_id"]
                .astype(int)
                .tolist()
            )
            cf_scores = raw_cf_scores.copy()
            for item_id in seen_items:
                cf_scores[self.product_index[item_id]] = 0.0
            self.cf_scores_raw_by_user[user_id] = raw_cf_scores
            self.cf_scores_by_user[user_id] = cf_scores.astype(np.float32)
            self.user_seen_items[user_id] = seen_items

        self._build_user_profiles(interactions)
        self._build_item_ctr(interactions)
        self.item_popularity = (
            interactions.groupby("product_id")["event_weight"]
            .sum()
            .sort_values(ascending=False)
            .index.astype(int)
            .tolist()
        )

    def _build_user_profiles(self, interactions: pd.DataFrame) -> None:
        merged = interactions.merge(self.products, on="product_id", how="left")
        category_scores: dict[int, dict[str, float]] = {}
        material_scores: dict[int, dict[str, float]] = {}
        style_scores: dict[int, dict[str, float]] = {}

        for user_id, user_frame in merged.groupby("user_id"):
            weights = user_frame["event_weight"].to_numpy(dtype=np.float32)
            item_indices = [self.product_index[int(item_id)] for item_id in user_frame["product_id"]]
            user_embeddings = self.embeddings[item_indices]
            weight_sum = float(weights.sum()) if len(weights) else 0.0
            if weight_sum > 0:
                profile = (user_embeddings * weights[:, None]).sum(axis=0) / weight_sum
            else:
                profile = np.zeros(self.embeddings.shape[1], dtype=np.float32)

            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
            self.user_profile_embeddings[int(user_id)] = profile.astype(np.float32)

            price_weights = user_frame["event_weight"].to_numpy(dtype=np.float32)
            prices = user_frame["price"].to_numpy(dtype=np.float32)
            self.user_avg_price[int(user_id)] = (
                float(np.average(prices, weights=price_weights))
                if len(prices)
                else float(self.products["price"].median())
            )

            category_scores[int(user_id)] = self._normalized_preference_map(
                user_frame, "category"
            )
            material_scores[int(user_id)] = self._normalized_preference_map(
                user_frame, "material"
            )
            style_scores[int(user_id)] = self._normalized_preference_map(user_frame, "style")

        global_avg_price = float(self.products["price"].median())
        for user_id in self.user_ids:
            self.user_profile_embeddings.setdefault(
                user_id, np.zeros(self.embeddings.shape[1], dtype=np.float32)
            )
            self.user_avg_price.setdefault(user_id, global_avg_price)
            self.user_pref_scores[user_id] = {
                "category": category_scores.get(user_id, {}),
                "material": material_scores.get(user_id, {}),
                "style": style_scores.get(user_id, {}),
            }

    def _normalized_preference_map(
        self,
        frame: pd.DataFrame,
        column: str,
    ) -> dict[str, float]:
        score_map = frame.groupby(column)["event_weight"].sum()
        total = float(score_map.sum())
        if total <= 0:
            return {}
        return {str(key): float(value / total) for key, value in score_map.items()}

    def _build_item_ctr(self, interactions: pd.DataFrame) -> None:
        event_counts = interactions.pivot_table(
            index="product_id",
            columns="event_type",
            values="user_id",
            aggfunc="count",
            fill_value=0,
        )
        for event_name in ["view", "wishlist", "cart", "purchase"]:
            if event_name not in event_counts.columns:
                event_counts[event_name] = 0

        views = event_counts["view"].astype(float)
        soft_clicks = (
            event_counts["wishlist"] * 0.35
            + event_counts["cart"] * 0.55
            + event_counts["purchase"] * 1.0
        )
        ctr = (soft_clicks + 1.0) / (views + event_counts.sum(axis=1) + 5.0)
        self.item_ctr = {int(item_id): float(score) for item_id, score in ctr.items()}

    def _normalize_scores(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value - min_value < 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - min_value) / (max_value - min_value)).astype(np.float32)

    def get_cf_scores(self, user_id: int, exclude_seen: bool = True) -> np.ndarray:
        score_store = self.cf_scores_by_user if exclude_seen else self.cf_scores_raw_by_user
        return score_store.get(int(user_id), np.zeros(len(self.item_ids), dtype=np.float32))

    def get_embedding_scores(self, user_id: int, query: str | None = None) -> np.ndarray:
        profile = self.user_profile_embeddings.get(
            int(user_id), np.zeros(self.embeddings.shape[1], dtype=np.float32)
        ).copy()
        if query:
            query_embedding = self.embedder.embed_query(query).astype(np.float32)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding /= query_norm
                profile = profile * 0.65 + query_embedding * 0.35
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile /= norm
        return (self.embeddings @ profile).astype(np.float32)

    def compute_feature_row(
        self,
        user_id: int,
        item_id: int,
        query: str | None = None,
        cf_score: float | None = None,
        embedding_score: float | None = None,
    ) -> dict[str, float | int | str]:
        product = self.products.iloc[self.product_index[int(item_id)]]
        cf_value = (
            float(cf_score)
            if cf_score is not None
            else float(
                self.get_cf_scores(user_id, exclude_seen=False)[self.product_index[item_id]]
            )
        )
        embed_value = (
            float(embedding_score)
            if embedding_score is not None
            else float(self.get_embedding_scores(user_id, query)[self.product_index[item_id]])
        )

        avg_price = self.user_avg_price.get(int(user_id), float(self.products["price"].median()))
        price_delta = abs(float(product["price"]) - avg_price)
        price_delta_from_avg = price_delta / max(avg_price, 1.0)
        price_sensitivity_match = max(0.0, 1.0 - price_delta / max(avg_price + 50.0, 1.0))

        user_prefs = self.user_pref_scores.get(int(user_id), {})
        category_match = user_prefs.get("category", {}).get(str(product["category"]), 0.0)
        material_match = user_prefs.get("material", {}).get(str(product["material"]), 0.0)
        style_match = user_prefs.get("style", {}).get(str(product["style"]), 0.0)
        historical_ctr = float(self.item_ctr.get(int(item_id), 0.0))

        feature_row = {
            "user_id": int(user_id),
            "product_id": int(item_id),
            "query": query or "",
            "cf_score": cf_value,
            "embedding_similarity": embed_value,
            "price_delta_from_user_avg": float(price_delta_from_avg),
            "price_sensitivity_match": float(price_sensitivity_match),
            "category_match_score": float(category_match),
            "material_preference_score": float(material_match),
            "style_affinity_score": float(style_match),
            "historical_ctr": historical_ctr,
            "product_price": float(product["price"]),
            "product_rating": float(product["rating"]),
        }
        feature_row["hybrid_score"] = self.hybrid_score(feature_row)
        return feature_row

    def hybrid_score(self, feature_row: dict[str, float | int | str]) -> float:
        score = 0.0
        for name, weight in self.blend_weights.items():
            score += weight * float(feature_row.get(name, 0.0))
        return float(score)

    def recommend_cf(self, user_id: int, top_k: int = 10) -> list[dict]:
        scores = self.get_cf_scores(user_id, exclude_seen=True)
        top_indices = np.argsort(scores)[::-1][:top_k]
        output = []
        for item_index in top_indices:
            item_id = self.item_ids[item_index]
            if scores[item_index] <= 0:
                continue
            output.append(
                {
                    "product_id": int(item_id),
                    "cf_score": round(float(scores[item_index]), 4),
                }
            )
        return output

    def recommend_content(self, user_id: int, query: str | None = None, top_k: int = 10) -> list[dict]:
        scores = self.get_embedding_scores(user_id, query)
        seen_items = self.user_seen_items.get(int(user_id), set())
        ranked_indices = np.argsort(scores)[::-1]
        output: list[dict] = []
        for item_index in ranked_indices:
            item_id = self.item_ids[item_index]
            if item_id in seen_items:
                continue
            if float(scores[item_index]) < self.similarity_threshold:
                continue
            output.append(
                {
                    "product_id": int(item_id),
                    "embedding_similarity": round(float(scores[item_index]), 4),
                }
            )
            if len(output) >= top_k:
                break
        return output

    def build_candidate_set(
        self,
        user_id: int,
        query: str | None = None,
        include_seen: bool = False,
    ) -> pd.DataFrame:
        cf_scores = self.get_cf_scores(user_id, exclude_seen=not include_seen)
        embedding_scores = self.get_embedding_scores(user_id, query)
        seen_items = self.user_seen_items.get(int(user_id), set())

        cf_top = [self.item_ids[idx] for idx in np.argsort(cf_scores)[::-1][: self.candidate_top_k]]
        content_top = [
            self.item_ids[idx]
            for idx in np.argsort(embedding_scores)[::-1]
            if float(embedding_scores[idx]) >= self.similarity_threshold
        ][: self.candidate_top_k]
        popularity_top = self.item_popularity[: max(10, self.candidate_top_k // 2)]

        candidate_ids: list[int] = []
        for item_id in cf_top + content_top + popularity_top:
            if not include_seen and item_id in seen_items:
                continue
            if item_id not in candidate_ids:
                candidate_ids.append(item_id)

        rows = [
            self.compute_feature_row(
                user_id=user_id,
                item_id=item_id,
                query=query,
                cf_score=float(cf_scores[self.product_index[item_id]]),
                embedding_score=float(embedding_scores[self.product_index[item_id]]),
            )
            for item_id in candidate_ids
        ]
        if not rows:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "query",
                    "cf_score",
                    "embedding_similarity",
                    "price_delta_from_user_avg",
                    "price_sensitivity_match",
                    "category_match_score",
                    "material_preference_score",
                    "style_affinity_score",
                    "historical_ctr",
                    "product_price",
                    "product_rating",
                    "hybrid_score",
                ]
            )
        return pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).reset_index(drop=True)

    def build_ltr_frame(
        self,
        user_ids: list[int],
        purchase_map: dict[int, set[int]],
        query_map: dict[int, str] | None = None,
    ) -> CandidateFrame:
        rows: list[pd.DataFrame] = []
        group_sizes: list[int] = []

        for user_id in user_ids:
            user_purchases = purchase_map.get(int(user_id), set())
            if not user_purchases:
                continue

            candidate_frame = self.build_candidate_set(
                user_id=user_id,
                query=(query_map or {}).get(int(user_id)),
                include_seen=True,
            )
            positive_rows = []
            for item_id in sorted(user_purchases):
                if int(item_id) in candidate_frame["product_id"].values:
                    continue
                positive_rows.append(
                    self.compute_feature_row(
                        user_id=user_id,
                        item_id=int(item_id),
                        query=(query_map or {}).get(int(user_id)),
                    )
                )
            if positive_rows:
                candidate_frame = pd.concat(
                    [candidate_frame, pd.DataFrame(positive_rows)],
                    ignore_index=True,
                )

            candidate_frame = candidate_frame.drop_duplicates("product_id").copy()
            candidate_frame["label"] = candidate_frame["product_id"].isin(user_purchases).astype(int)
            candidate_frame = candidate_frame.sort_values(
                ["label", "hybrid_score"],
                ascending=[False, False],
            ).reset_index(drop=True)

            if candidate_frame["label"].sum() == 0 or candidate_frame["label"].nunique() < 2:
                continue

            rows.append(candidate_frame)
            group_sizes.append(int(len(candidate_frame)))

        features = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return CandidateFrame(features=features, group_sizes=group_sizes)

    def hybrid_recommend(
        self,
        user_id: int,
        query: str | None = None,
        top_k: int = 5,
        ltr_model: object | None = None,
    ) -> list[dict]:
        candidates = self.build_candidate_set(user_id=user_id, query=query, include_seen=False)
        if candidates.empty:
            return []

        rank_frame = candidates.copy()
        feature_columns = [
            "cf_score",
            "embedding_similarity",
            "price_delta_from_user_avg",
            "price_sensitivity_match",
            "category_match_score",
            "material_preference_score",
            "style_affinity_score",
            "historical_ctr",
            "product_price",
            "product_rating",
            "hybrid_score",
        ]
        if ltr_model is not None:
            rank_frame["ltr_score"] = ltr_model.predict(rank_frame[feature_columns])
            rank_frame = rank_frame.sort_values(
                ["ltr_score", "hybrid_score"], ascending=False
            ).reset_index(drop=True)
        else:
            rank_frame["ltr_score"] = rank_frame["hybrid_score"]

        output = []
        for row in rank_frame.head(top_k).itertuples(index=False):
            product = self.products.iloc[self.product_index[int(row.product_id)]]
            output.append(
                {
                    "product_id": int(row.product_id),
                    "name": str(product["name"]),
                    "category": str(product["category"]),
                    "material": str(product["material"]),
                    "style": str(product["style"]),
                    "price": float(product["price"]),
                    "rating": float(product["rating"]),
                    "cf_score": round(float(row.cf_score), 4),
                    "embedding_similarity": round(float(row.embedding_similarity), 4),
                    "hybrid_score": round(float(row.hybrid_score), 4),
                    "ltr_score": round(float(row.ltr_score), 4),
                }
            )
        return output

    def save_logic_summary(self) -> Path:
        output_path = self.project_root / "experiments" / "hybrid_logic_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signal_sources": [
                "Collaborative Filtering score",
                "Embedding similarity score",
                "Price sensitivity match",
                "Style preference match",
                "Historical CTR (simulated)",
            ],
            "blend_weights": self.blend_weights,
            "candidate_top_k": self.candidate_top_k,
            "similarity_threshold": self.similarity_threshold,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
