from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.svmx.data.preprocess import preprocess
from src.svmx.models.registry import build_model
from src.svmx.models.train import fit_model, evaluate_model
from src.svmx.explainers.svmx import SVMXExplainer
from src.svmx.evaluation.fidelity import evaluate_fidelity, evaluate_weight_stability
from src.svmx.utils.seed import set_seed


def make_synthetic_adult(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "fnlwgt": rng.randint(10000, 500000, n),
        "education-num": rng.randint(1, 17, n),
        "capital-gain": rng.randint(0, 100000, n),
        "capital-loss": rng.randint(0, 5000, n),
        "hours-per-week": rng.randint(1, 99, n),
        "workclass": rng.choice(["Private", "Self-emp", "Gov"], n),
        "education": rng.choice(["Bachelors", "Masters", "HS-grad"], n),
        "marital-status": rng.choice(["Married", "Single", "Divorced"], n),
        "occupation": rng.choice(["Tech", "Sales", "Craft"], n),
        "relationship": rng.choice(["Husband", "Wife", "Own-child"], n),
        "race": rng.choice(["White", "Black", "Asian"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "native-country": rng.choice(["US", "Mexico", "India"], n),
        "income": rng.choice(["<=50K", ">50K"], n),
    })


def make_synthetic_bank(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "balance": rng.randint(-5000, 100000, n),
        "day": rng.randint(1, 31, n),
        "duration": rng.randint(0, 4000, n),
        "campaign": rng.randint(1, 50, n),
        "pdays": rng.choice([-1, 30, 90, 180], n),
        "previous": rng.randint(0, 10, n),
        "job": rng.choice(["admin.", "technician", "services"], n),
        "marital": rng.choice(["married", "single", "divorced"], n),
        "education": rng.choice(["primary", "secondary", "tertiary"], n),
        "default": rng.choice(["yes", "no"], n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "telephone", "unknown"], n),
        "month": rng.choice(["jan", "feb", "mar", "apr"], n),
        "poutcome": rng.choice(["success", "failure", "unknown"], n),
        "y": rng.choice(["yes", "no"], n),
    })


def load_demo_dataframe(dataset_name: str, seed: int) -> pd.DataFrame:
    if dataset_name == "adult":
        return make_synthetic_adult(seed=seed)
    if dataset_name == "bank":
        return make_synthetic_bank(seed=seed)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal SVM-X explanation experiment.")
    parser.add_argument("--dataset", choices=["adult", "bank"], required=True)
    parser.add_argument("--model", choices=["rf", "lr", "dt", "xgb"], required=True)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--target_index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # 1) build dataset
    df = load_demo_dataframe(args.dataset, seed=args.seed)

    # 2) split raw dataframe first
    df_train, df_test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["income"] if args.dataset == "adult" else df["y"],
    )

    # 3) preprocess train/test with consistent columns
    X_train, y_train, stats_train, scaler = preprocess(
        df_train,
        dataset_name=args.dataset,
        fit_scaler=True,
    )
    X_test, y_test, _, _ = preprocess(
        df_test,
        dataset_name=args.dataset,
        scaler=scaler,
        fit_scaler=False,
        expected_columns=stats_train["feature_names"],
    )

    # 4) train target model
    model = build_model(args.model, random_state=args.seed)
    model = fit_model(model, X_train, y_train)
    target_metrics = evaluate_model(model, X_test, y_test)

    # 5) explain one target instance
    target_index = min(args.target_index, len(X_test) - 1)
    x_t = X_test[target_index]

    explainer = SVMXExplainer(
        n_samples=args.n_samples,
        top_k=args.top_k,
        random_state=args.seed,
    )

    result = explainer.explain(
        target_record=x_t,
        predict_fn=model.predict,
        predict_proba_fn=model.predict_proba,
        feature_stats=stats_train,
    )

    # 6) compute local fidelity
    fidelity = evaluate_fidelity(
        neighbours=result["neighbours"],
        predict_fn=model.predict,
        surrogate_predict_fn=explainer.predict_surrogate,
        predict_proba_fn=model.predict_proba,
        surrogate_proba_fn=explainer.predict_proba_surrogate,
    )

    # 7) compute stability on same target
    stability = evaluate_weight_stability(
        target_record=x_t,
        explain_fn=lambda record: explainer.explain(
            target_record=record,
            predict_fn=model.predict,
            predict_proba_fn=model.predict_proba,
            feature_stats=stats_train,
        )["all_weights"],
        n_neighbours=30,
        noise_scale=0.01,
        random_state=args.seed,
    )

    # 8) save JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "top_k": args.top_k,
        "target_index": int(target_index),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "feature_count": int(X_train.shape[1]),
        "target_model_metrics": target_metrics,
        "fidelity_metrics": fidelity,
        "weight_stability": stability,
        "top_k_indices": result["top_k_indices"].tolist(),
        "top_k_weights": result["top_k_weights"].tolist(),
        "feature_names": stats_train["feature_names"],
    }

    output_path = output_dir / f"{args.dataset}_{args.model}_seed{args.seed}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] Saved results to: {output_path}")
    print(json.dumps(payload["fidelity_metrics"], indent=2))


if __name__ == "__main__":
    main()
