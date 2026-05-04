from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from models.hybrid_recommender import HybridRecommender
from models.ltr_model import LTRTrainer

def write_notebook(project_root: Path) -> Path:
    notebook_path = project_root / "notebooks" / "hybrid_analysis.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Hybrid Recommender Analysis\n",
                    "\n",
                    "This notebook loads the Week 4 experiment outputs and shows the core hybrid ranking artifacts.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "project_root = Path.cwd().resolve().parents[0] if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n",
                    "result_path = project_root / 'experiments' / 'experiment_results.json'\n",
                    "results = json.loads(result_path.read_text(encoding='utf-8'))\n",
                    "results.keys()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "results['baseline_results']\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "results['feature_importance']\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return notebook_path


def write_readme(project_root: Path, summary: dict[str, object]) -> Path:
    readme_path = project_root / "README.md"
    baseline = summary["baseline_results"]
    trial_count = summary["optuna_trial_count"]
    ndcg10_vs_cf = summary["ndcg10_improvement_vs_cf_pct"]
    readme_text = f"""# Week 4 Hybrid Recommender

This folder builds a hybrid recommender using the local project assets in `data/` and `src/` with:

- collaborative filtering from weighted user-item interactions
- embedding-based retrieval using the provided product embeddings
- hand-engineered hybrid scoring logic
- XGBoost learn-to-rank reranking
- Optuna hyperparameter tuning
- held-out evaluation against CF and content baselines

## Structure

- `models/hybrid_recommender.py`: hybrid signal computation and `hybrid_recommend`.
- `models/ltr_model.py`: feature matrix creation, XGBoost ranker training, Optuna tuning, and evaluation.
- `models/model_checkpoints/`: trained model and feature-importance artifacts.
- `experiments/optuna_study.db`: Optuna SQLite study.
- `experiments/experiment_results.json`: tracked metrics and best parameters.
- `evaluation/hybrid_evaluation_report.pdf`: PDF report.
- `notebooks/hybrid_analysis.ipynb`: analysis notebook scaffold.

## Hybrid Logic

Signals used:

- Collaborative Filtering score
- Embedding similarity score
- Price sensitivity match
- Style preference match
- Historical CTR (simulated)

The feature matrix includes:

- `cf_score`
- `embedding_similarity`
- `price_delta_from_user_avg`
- `category_match_score`
- `material_preference_score`
- `style_affinity_score`
- `price_sensitivity_match`
- `historical_ctr`
- `product_price`
- `product_rating`
- `hybrid_score`

## Training and Evaluation

The LTR model is trained with `XGBRanker(objective='rank:pairwise')` on user-level grouped candidate sets. Users with at least one purchase are split into train/validation/test groups with a fixed random seed.

Latest test-set results:

- `cf_only`: Precision@5={baseline['cf_only']['precision@5']}, Precision@10={baseline['cf_only']['precision@10']}, NDCG@5={baseline['cf_only']['ndcg@5']}, NDCG@10={baseline['cf_only']['ndcg@10']}
- `content_only`: Precision@5={baseline['content_only']['precision@5']}, Precision@10={baseline['content_only']['precision@10']}, NDCG@5={baseline['content_only']['ndcg@5']}, NDCG@10={baseline['content_only']['ndcg@10']}
- `hybrid_weighted`: Precision@5={baseline['hybrid_weighted']['precision@5']}, Precision@10={baseline['hybrid_weighted']['precision@10']}, NDCG@5={baseline['hybrid_weighted']['ndcg@5']}, NDCG@10={baseline['hybrid_weighted']['ndcg@10']}
- `hybrid_ltr`: Precision@5={baseline['hybrid_ltr']['precision@5']}, Precision@10={baseline['hybrid_ltr']['precision@10']}, NDCG@5={baseline['hybrid_ltr']['ndcg@5']}, NDCG@10={baseline['hybrid_ltr']['ndcg@10']}

NDCG@10 improvement vs CF baseline: {ndcg10_vs_cf}%

Optuna study trial count: {trial_count}

Best Optuna parameters are stored in `experiments/experiment_results.json`.

## Run

From this folder:

```powershell
.\\venv\\Scripts\\python.exe run_week4.py
```
"""
    readme_path.write_text(readme_text, encoding="utf-8")
    return readme_path


def zip_project(project_root: Path) -> Path:
    zip_path = project_root / "week4_submission.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in project_root.rglob("*"):
            if path == zip_path or path.is_dir():
                continue
            if "__pycache__" in path.parts or "venv" in path.parts:
                continue
            archive.write(path, arcname=path.relative_to(project_root))
    return zip_path

def main() -> None:
    project_root = Path(__file__).resolve().parent

    trainer = LTRTrainer(project_root=project_root)
    trainer.recommender.save_logic_summary()

    frames = trainer.build_frames()
    best_params, study = trainer.tune(frames, n_trials=50)

    tuned_weights = {
        "cf_score": best_params["w_cf"],
        "embedding_similarity": best_params["w_embed"],
        "price_sensitivity_match": best_params["w_price"],
        "style_affinity_score": best_params["w_style"],
        "material_preference_score": best_params["w_material"],
        "category_match_score": best_params["w_category"],
        "historical_ctr": best_params["w_ctr"],
    }
    weight_sum = sum(tuned_weights.values())
    tuned_weights = {key: value / weight_sum for key, value in tuned_weights.items()}

    trainer = LTRTrainer(
        project_root=project_root,
        candidate_top_k=int(best_params["candidate_top_k"]),
        blend_weights=tuned_weights,
        similarity_threshold=float(best_params["similarity_threshold"]),
    )
    frames = trainer.build_frames()
    ranker, test_metrics = trainer.train_final_model(frames, best_params)
    baseline_results = trainer.evaluate_baselines(
        frames["test"].features, frames["test"].group_sizes, ranker
    )
    artifact_paths = trainer.save_artifacts(
        ranker=ranker,
        best_params=best_params,
        study=study,
        baseline_results=baseline_results,
        test_metrics=test_metrics,
        frames=frames,
    )

    example_recommendations = trainer.recommender.hybrid_recommend(
        user_id=trainer.split.test_users[0],
        query=trainer.split.query_map[trainer.split.test_users[0]],
        top_k=5,
        ltr_model=ranker,
    )
    sample_path = project_root / "experiments" / "sample_recommendations.json"
    sample_path.write_text(json.dumps(example_recommendations, indent=2), encoding="utf-8")

    write_notebook(project_root)
    result_payload = json.loads(
        (project_root / "experiments" / "experiment_results.json").read_text(encoding="utf-8")
    )
    readme_path = write_readme(
        project_root,
        {
            "baseline_results": baseline_results,
            "test_metrics": test_metrics,
            "optuna_trial_count": result_payload["optuna_trial_count"],
            "ndcg10_improvement_vs_cf_pct": result_payload["ndcg10_improvement_vs_cf_pct"],
        },
    )
    zip_path = zip_project(project_root)

    summary = {
        "artifacts": artifact_paths,
        "readme_path": str(readme_path),
        "zip_path": str(zip_path),
        "sample_recommendations_path": str(sample_path),
        "test_metrics": test_metrics,
        "baseline_results": baseline_results,
        "optuna_trial_count": result_payload["optuna_trial_count"],
        "ndcg10_improvement_vs_cf_pct": result_payload["ndcg10_improvement_vs_cf_pct"],
    }
    output_path = project_root / "experiments" / "run_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
