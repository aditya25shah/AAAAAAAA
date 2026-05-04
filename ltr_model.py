from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from xgboost import XGBRanker

from models.hybrid_recommender import HybridRecommender

FEATURE_COLUMNS = [
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


@dataclass
class DatasetSplit:
    train_users: list[int]
    val_users: list[int]
    test_users: list[int]
    purchase_map: dict[int, set[int]]
    query_map: dict[int, str]


def precision_at_k(labels: list[int], k: int) -> float:
    if not labels:
        return 0.0
    top = labels[:k]
    return float(sum(top) / max(k, 1))


def ndcg_at_k(labels: list[int], k: int) -> float:
    top = labels[:k]
    if not top:
        return 0.0
    dcg = sum((2**label - 1) / np.log2(idx + 2) for idx, label in enumerate(top))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum((2**label - 1) / np.log2(idx + 2) for idx, label in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


class LTRTrainer:
    def __init__(
        self,
        project_root: Path | None = None,
        candidate_top_k: int = 35,
        blend_weights: dict[str, float] | None = None,
        similarity_threshold: float = 0.18,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.models_dir = self.project_root / "models"
        self.checkpoint_dir = self.models_dir / "model_checkpoints"
        self.experiments_dir = self.project_root / "experiments"
        self.evaluation_dir = self.project_root / "evaluation"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

        self.recommender = HybridRecommender(
            project_root=self.project_root,
            candidate_top_k=candidate_top_k,
            blend_weights=blend_weights,
            similarity_threshold=similarity_threshold,
        )
        self.split = self._build_split()

    def _build_split(self) -> DatasetSplit:
        purchases = (
            self.recommender.behavior.loc[
                self.recommender.behavior["event_type"] == "purchase",
                ["user_id", "product_id", "timestamp"],
            ]
            .sort_values(["user_id", "timestamp"])
            .copy()
        )
        purchase_map = {
            int(user_id): set(frame["product_id"].astype(int).tolist())
            for user_id, frame in purchases.groupby("user_id")
        }

        eligible_users = sorted(purchase_map.keys())
        rng = np.random.default_rng(42)
        shuffled = rng.permutation(eligible_users).tolist()
        n_users = len(shuffled)
        train_end = int(n_users * 0.7)
        val_end = int(n_users * 0.85)

        query_map: dict[int, str] = {}
        product_lookup = self.recommender.products.set_index("product_id")
        for user_id, item_ids in purchase_map.items():
            top_item = int(sorted(item_ids)[0])
            product = product_lookup.loc[top_item]
            query_map[user_id] = (
                f"{product['style']} {product['category']} {product['material']} under {int(product['price']) + 100}"
            )

        return DatasetSplit(
            train_users=[int(value) for value in shuffled[:train_end]],
            val_users=[int(value) for value in shuffled[train_end:val_end]],
            test_users=[int(value) for value in shuffled[val_end:]],
            purchase_map=purchase_map,
            query_map=query_map,
        )

    def build_frames(self) -> dict[str, object]:
        train = self.recommender.build_ltr_frame(
            user_ids=self.split.train_users,
            purchase_map=self.split.purchase_map,
            query_map=self.split.query_map,
        )
        val = self.recommender.build_ltr_frame(
            user_ids=self.split.val_users,
            purchase_map=self.split.purchase_map,
            query_map=self.split.query_map,
        )
        test = self.recommender.build_ltr_frame(
            user_ids=self.split.test_users,
            purchase_map=self.split.purchase_map,
            query_map=self.split.query_map,
        )
        return {"train": train, "val": val, "test": test}

    def _make_ranker(self, params: dict[str, float | int]) -> XGBRanker:
        return XGBRanker(
            objective="rank:pairwise",
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            n_estimators=int(params["n_estimators"]),
            random_state=42,
            eval_metric="ndcg@5",
            tree_method="hist",
        )

    def _evaluate_ranker(
        self,
        frame: pd.DataFrame,
        group_sizes: list[int],
        ranker: XGBRanker,
    ) -> dict[str, float]:
        if frame.empty:
            return {
                "precision@5": 0.0,
                "precision@10": 0.0,
                "ndcg@5": 0.0,
                "ndcg@10": 0.0,
                "diversity@5": 0.0,
                "diversity@10": 0.0,
            }

        scored = frame.copy()
        scored["score"] = ranker.predict(scored[FEATURE_COLUMNS])

        precisions_at_5: list[float] = []
        precisions_at_10: list[float] = []
        ndcgs_at_5: list[float] = []
        ndcgs_at_10: list[float] = []
        diversities_at_5: list[float] = []
        diversities_at_10: list[float] = []
        category_lookup = self.recommender.products.set_index("product_id")["category"]
        start = 0
        for group_size in group_sizes:
            group = scored.iloc[start : start + group_size].sort_values("score", ascending=False)
            labels = group["label"].astype(int).tolist()
            precisions_at_5.append(precision_at_k(labels, 5))
            precisions_at_10.append(precision_at_k(labels, 10))
            ndcgs_at_5.append(ndcg_at_k(labels, 5))
            ndcgs_at_10.append(ndcg_at_k(labels, 10))
            top_categories_5 = group.head(5)["product_id"].map(category_lookup)
            top_categories_10 = group.head(10)["product_id"].map(category_lookup)
            diversities_at_5.append(
                float(top_categories_5.nunique() / max(len(top_categories_5), 1))
            )
            diversities_at_10.append(
                float(top_categories_10.nunique() / max(len(top_categories_10), 1))
            )
            start += group_size

        return {
            "precision@5": round(float(np.mean(precisions_at_5)), 4),
            "precision@10": round(float(np.mean(precisions_at_10)), 4),
            "ndcg@5": round(float(np.mean(ndcgs_at_5)), 4),
            "ndcg@10": round(float(np.mean(ndcgs_at_10)), 4),
            "diversity@5": round(float(np.mean(diversities_at_5)), 4),
            "diversity@10": round(float(np.mean(diversities_at_10)), 4),
        }

    def tune(self, frames: dict[str, object], n_trials: int = 50) -> tuple[dict, optuna.Study]:
        study_path = self.experiments_dir / "optuna_study.db"
        storage = f"sqlite:///{study_path.as_posix()}"

        def objective(trial: optuna.Trial) -> float:
            blend_weights = {
                "cf_score": trial.suggest_float("w_cf", 0.2, 0.5),
                "embedding_similarity": trial.suggest_float("w_embed", 0.15, 0.4),
                "price_sensitivity_match": trial.suggest_float("w_price", 0.05, 0.2),
                "style_affinity_score": trial.suggest_float("w_style", 0.05, 0.15),
                "material_preference_score": trial.suggest_float("w_material", 0.03, 0.12),
                "category_match_score": trial.suggest_float("w_category", 0.03, 0.12),
                "historical_ctr": trial.suggest_float("w_ctr", 0.02, 0.12),
            }
            weight_sum = sum(blend_weights.values())
            blend_weights = {key: value / weight_sum for key, value in blend_weights.items()}

            tuned_recommender = HybridRecommender(
                project_root=self.project_root,
                candidate_top_k=trial.suggest_int("candidate_top_k", 25, 45),
                blend_weights=blend_weights,
                similarity_threshold=trial.suggest_float("similarity_threshold", 0.1, 0.35),
            )
            train = tuned_recommender.build_ltr_frame(
                user_ids=self.split.train_users,
                purchase_map=self.split.purchase_map,
                query_map=self.split.query_map,
            )
            val = tuned_recommender.build_ltr_frame(
                user_ids=self.split.val_users,
                purchase_map=self.split.purchase_map,
                query_map=self.split.query_map,
            )
            if train.features.empty or val.features.empty:
                return 0.0

            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.18),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 5.0),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
                "n_estimators": trial.suggest_int("n_estimators", 80, 220),
            }
            ranker = self._make_ranker(params)
            ranker.fit(
                train.features[FEATURE_COLUMNS],
                train.features["label"],
                group=train.group_sizes,
            )
            metrics = self._evaluate_ranker(val.features, val.group_sizes, ranker)
            return metrics["ndcg@10"]

        study = optuna.create_study(
            study_name="hybrid_ltr_optimization",
            direction="maximize",
            storage=storage,
            load_if_exists=True,
        )
        remaining = max(0, n_trials - len(study.trials))
        if remaining:
            study.optimize(objective, n_trials=remaining)

        best = dict(study.best_params)
        return best, study

    def train_final_model(
        self,
        frames: dict[str, object],
        best_params: dict[str, float | int],
    ) -> tuple[XGBRanker, dict[str, float]]:
        train_plus_val = pd.concat(
            [frames["train"].features, frames["val"].features],
            ignore_index=True,
        )
        train_groups = frames["train"].group_sizes + frames["val"].group_sizes

        model_params = {
            "learning_rate": best_params["learning_rate"],
            "max_depth": best_params["max_depth"],
            "min_child_weight": best_params["min_child_weight"],
            "subsample": best_params["subsample"],
            "colsample_bytree": best_params["colsample_bytree"],
            "reg_alpha": best_params["reg_alpha"],
            "reg_lambda": best_params["reg_lambda"],
            "n_estimators": best_params["n_estimators"],
        }
        ranker = self._make_ranker(model_params)
        ranker.fit(
            train_plus_val[FEATURE_COLUMNS],
            train_plus_val["label"],
            group=train_groups,
        )
        metrics = self._evaluate_ranker(frames["test"].features, frames["test"].group_sizes, ranker)
        return ranker, metrics

    def evaluate_baselines(
        self,
        test_frame: pd.DataFrame,
        test_groups: list[int],
        ranker: XGBRanker,
    ) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        baseline_map = {
            "cf_only": "cf_score",
            "content_only": "embedding_similarity",
            "hybrid_weighted": "hybrid_score",
        }
        for name, score_column in baseline_map.items():
            scored = test_frame.copy()
            scored["score"] = scored[score_column]
            results[name] = self._group_metric_summary(scored, test_groups, "score")

        reranked = test_frame.copy()
        reranked["score"] = ranker.predict(reranked[FEATURE_COLUMNS])
        results["hybrid_ltr"] = self._group_metric_summary(reranked, test_groups, "score")
        return results

    def _group_metric_summary(
        self,
        frame: pd.DataFrame,
        group_sizes: list[int],
        score_column: str,
    ) -> dict[str, float]:
        precisions_at_5: list[float] = []
        precisions_at_10: list[float] = []
        ndcgs_at_5: list[float] = []
        ndcgs_at_10: list[float] = []
        diversities_at_5: list[float] = []
        diversities_at_10: list[float] = []
        start = 0
        category_lookup = self.recommender.products.set_index("product_id")["category"]
        for group_size in group_sizes:
            group = frame.iloc[start : start + group_size].sort_values(score_column, ascending=False)
            labels = group["label"].astype(int).tolist()
            precisions_at_5.append(precision_at_k(labels, 5))
            precisions_at_10.append(precision_at_k(labels, 10))
            ndcgs_at_5.append(ndcg_at_k(labels, 5))
            ndcgs_at_10.append(ndcg_at_k(labels, 10))
            top_categories_5 = group.head(5)["product_id"].map(category_lookup)
            top_categories_10 = group.head(10)["product_id"].map(category_lookup)
            diversities_at_5.append(
                float(top_categories_5.nunique() / max(len(top_categories_5), 1))
            )
            diversities_at_10.append(
                float(top_categories_10.nunique() / max(len(top_categories_10), 1))
            )
            start += group_size
        return {
            "precision@5": round(float(np.mean(precisions_at_5)), 4),
            "precision@10": round(float(np.mean(precisions_at_10)), 4),
            "ndcg@5": round(float(np.mean(ndcgs_at_5)), 4),
            "ndcg@10": round(float(np.mean(ndcgs_at_10)), 4),
            "diversity@5": round(float(np.mean(diversities_at_5)), 4),
            "diversity@10": round(float(np.mean(diversities_at_10)), 4),
        }

    def save_artifacts(
        self,
        ranker: XGBRanker,
        best_params: dict[str, float | int],
        study: optuna.Study,
        baseline_results: dict[str, dict[str, float]],
        test_metrics: dict[str, float],
        frames: dict[str, object],
    ) -> dict[str, str]:
        model_path = self.checkpoint_dir / "xgb_ltr_model.json"
        feature_path = self.checkpoint_dir / "feature_importance.json"
        metadata_path = self.checkpoint_dir / "training_metadata.json"
        result_path = self.experiments_dir / "experiment_results.json"
        report_path = self.evaluation_dir / "hybrid_evaluation_report.pdf"

        ranker.save_model(model_path)
        feature_importance = {
            feature: float(score)
            for feature, score in zip(FEATURE_COLUMNS, ranker.feature_importances_)
        }
        feature_path.write_text(json.dumps(feature_importance, indent=2), encoding="utf-8")

        metadata = {
            "feature_columns": FEATURE_COLUMNS,
            "train_rows": int(len(frames["train"].features)),
            "val_rows": int(len(frames["val"].features)),
            "test_rows": int(len(frames["test"].features)),
            "train_groups": len(frames["train"].group_sizes),
            "val_groups": len(frames["val"].group_sizes),
            "test_groups": len(frames["test"].group_sizes),
            "best_params": best_params,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        trials_payload = []
        for trial in study.trials:
            if trial.value is None:
                continue
            trials_payload.append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                }
            )
        result_payload = {
            "best_params": best_params,
            "feature_importance": feature_importance,
            "test_metrics": test_metrics,
            "baseline_results": baseline_results,
            "optuna_trial_count": len(trials_payload),
            "ndcg10_improvement_vs_cf_pct": round(
                (
                    (baseline_results["hybrid_ltr"]["ndcg@10"] - baseline_results["cf_only"]["ndcg@10"])
                    / max(baseline_results["cf_only"]["ndcg@10"], 1e-8)
                )
                * 100.0,
                2,
            ),
            "ndcg10_improvement_vs_content_pct": round(
                (
                    (
                        baseline_results["hybrid_ltr"]["ndcg@10"]
                        - baseline_results["content_only"]["ndcg@10"]
                    )
                    / max(baseline_results["content_only"]["ndcg@10"], 1e-8)
                )
                * 100.0,
                2,
            ),
            "optuna_trials": trials_payload,
        }
        result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

        self._write_pdf_report(report_path, baseline_results, best_params, feature_importance)
        return {
            "model_path": str(model_path),
            "feature_path": str(feature_path),
            "metadata_path": str(metadata_path),
            "result_path": str(result_path),
            "report_path": str(report_path),
        }

    def _write_pdf_report(
        self,
        path: Path,
        baseline_results: dict[str, dict[str, float]],
        best_params: dict[str, float | int],
        feature_importance: dict[str, float],
    ) -> None:
        pdf = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        y = height - 50

        def line(text: str) -> None:
            nonlocal y
            pdf.drawString(50, y, text)
            y -= 18
            if y < 70:
                pdf.showPage()
                y = height - 50

        pdf.setFont("Helvetica-Bold", 16)
        line("Hybrid Recommender Evaluation Report")
        pdf.setFont("Helvetica", 11)
        line("")
        line("Baselines vs Hybrid LTR")
        for model_name, metrics in baseline_results.items():
            line(
                f"{model_name}: Precision@5={metrics['precision@5']}, Precision@10={metrics['precision@10']}, "
                f"NDCG@5={metrics['ndcg@5']}, NDCG@10={metrics['ndcg@10']}, "
                f"Diversity@5={metrics['diversity@5']}, Diversity@10={metrics['diversity@10']}"
            )
        cf_ndcg10 = baseline_results["cf_only"]["ndcg@10"]
        hybrid_ndcg10 = baseline_results["hybrid_ltr"]["ndcg@10"]
        cf_improvement = ((hybrid_ndcg10 - cf_ndcg10) / max(cf_ndcg10, 1e-8)) * 100.0
        line("")
        line(f"NDCG@10 improvement vs CF baseline: {round(cf_improvement, 2)}%")
        line("")
        line("Best Parameters")
        for key, value in best_params.items():
            line(f"{key}: {value}")
        line("")
        line("Feature Importance")
        for key, value in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True):
            line(f"{key}: {round(value, 4)}")

        pdf.save()
