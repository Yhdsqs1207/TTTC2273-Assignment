from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from model import (  # noqa: E402
    FEATURES,
    RANGES,
    EXPERT_GENOME,
    EXPERT_KNOTS,
    generate_dataset,
    predict,
    objectives,
    decode,
    memberships,
    normalize_X,
    rule_texts,
    coverage_overlap_check,
)
from optimizers import nsga2, knee_index, scalar_ga, random_search, nondominated_sort, dense_accuracy_ga  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
IMG_DIR = ROOT / "images"
for d in [DATA_DIR, RESULTS_DIR, IMG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N = 420
DATA_SEED = 20260909
NOISE_SD = 2.2
POP_SIZE = 24
GENERATIONS = 12
EVAL_BUDGET = POP_SIZE * (GENERATIONS + 1)
N_SEEDS = 30
TRAIN_FRAC = 0.70
LAMBDA = 0.10


def poly_features(X):
    Xn = normalize_X(X)
    cols = [np.ones(len(X))]
    cols.extend([Xn[:, j] for j in range(4)])
    cols.extend([Xn[:, j] ** 2 for j in range(4)])
    for j in range(4):
        for k in range(j + 1, 4):
            cols.append(Xn[:, j] * Xn[:, k])
    ph = RANGES[1, 0] + Xn[:, 1] * (RANGES[1, 1] - RANGES[1, 0])
    cols.append(np.abs(ph - 7.0) / 1.5)
    cols.append(np.sqrt(np.clip(Xn[:, 0], 0.0, None)))
    return np.column_stack(cols)


def fit_ridge(X, y, alpha=1e-3):
    A = poly_features(X)
    reg = np.eye(A.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + alpha * reg, A.T @ y)


def ridge_predict(X, beta):
    return poly_features(X) @ beta


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def save_membership_plots(genome):
    knots, _, _ = decode(genome)
    labels = ["Low", "Medium", "High"]
    feature_titles = [
        "Raw-water turbidity (NTU)",
        "pH",
        "Inflow rate (MLD)",
        "12-hour rainfall (mm)",
    ]
    for j, title in enumerate(feature_titles):
        x_actual = np.linspace(RANGES[j, 0], RANGES[j, 1], 401)
        Xdummy = np.tile(np.array([30.0, 7.0, 65.0, 10.0]), (401, 1))
        Xdummy[:, j] = x_actual
        mu = memberships(normalize_X(Xdummy), knots)[:, j, :]
        plt.figure(figsize=(6.4, 3.5))
        for k in range(3):
            plt.plot(x_actual, mu[:, k], label=labels[k], linewidth=2)
        plt.xlabel(title)
        plt.ylabel("Membership")
        plt.ylim(-0.02, 1.04)
        plt.legend(frameon=False, ncol=3)
        plt.tight_layout()
        plt.savefig(IMG_DIR / f"membership_{j+1}.png", dpi=180)
        plt.close()


def main():
    df = generate_dataset(N, DATA_SEED, NOISE_SD)
    df.to_csv(DATA_DIR / "synthetic_water_treatment.csv", index=False)
    X = df[FEATURES].values
    y = df["ideal_alum_dose_mg_L"].values

    # 200 randomized candidates across complexity levels. First obtain a dense
    # accuracy anchor, then randomly prune optional rules and perturb parameters.
    # This diagnostic isolates the intended accuracy-vs-simplicity tension rather
    # than mixing it with obviously bad random partitions.
    dense_anchor, dense_anchor_obj = dense_accuracy_ga(X, y, seed=314159, pop_size=40, generations=50)
    rng = np.random.default_rng(271828)
    random_rows = []
    for i in range(200):
        g = dense_anchor.copy()
        keep_probability = rng.uniform(0.05, 1.0)
        g[48:] = (rng.random(27) < keep_probability).astype(float)
        g[:12] += rng.normal(0.0, 0.015, 12)
        g[12:48] += rng.normal(0.0, 1.0, 36)
        g = np.clip(g, np.r_[np.full(12, 0.08), np.full(36, 8.0), np.zeros(27)], np.r_[np.full(12, 0.92), np.full(36, 86.0), np.ones(27)])
        o = objectives(X, y, g)
        random_rows.append({"candidate": i + 1, "rmse_mg_L": o[0], "interpretability_cost": o[1], "optional_keep_probability": keep_probability})
    random_df = pd.DataFrame(random_rows)
    random_df.to_csv(RESULTS_DIR / "random_200_candidates.csv", index=False)
    rho, rho_p = spearmanr(random_df["rmse_mg_L"], random_df["interpretability_cost"])

    plt.figure(figsize=(6.2, 4.0))
    plt.scatter(random_df["interpretability_cost"], random_df["rmse_mg_L"], s=24, alpha=0.68)
    plt.xlabel("Interpretability cost (lower = simpler)")
    plt.ylabel("RMSE (mg/L, lower = better)")
    plt.title("200 random candidates: accuracy-interpretability conflict")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "random_conflict_scatter.png", dpi=190)
    plt.close()

    rows = []
    selected_genomes = []
    n_train = int(TRAIN_FRAC * N)
    for seed in range(N_SEEDS):
        split_rng = np.random.default_rng(10_000 + seed)
        order = split_rng.permutation(N)
        tr, te = order[:n_train], order[n_train:]
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

        # 1) Untuned expert fuzzy rules
        expert_pred, _ = predict(Xte, EXPERT_GENOME)
        rows.append({"seed": seed, "method": "Untuned expert fuzzy", "test_rmse": rmse(yte, expert_pred), "interpretability_cost": objectives(Xtr, ytr, EXPERT_GENOME)[1]})

        # 2) Random search, same evaluation budget
        rg, _ = random_search(Xtr, ytr, seed=20_000 + seed, budget=EVAL_BUDGET, lambda_interpretability=LAMBDA)
        rp, _ = predict(Xte, rg)
        rows.append({"seed": seed, "method": "Random search", "test_rmse": rmse(yte, rp), "interpretability_cost": objectives(Xtr, ytr, rg)[1]})

        # 3) Scalarised single-objective GA, same evaluation budget
        gg, _ = scalar_ga(Xtr, ytr, seed=30_000 + seed, pop_size=POP_SIZE, generations=GENERATIONS, lambda_interpretability=LAMBDA)
        gp, _ = predict(Xte, gg)
        rows.append({"seed": seed, "method": "Scalarised GA", "test_rmse": rmse(yte, gp), "interpretability_cost": objectives(Xtr, ytr, gg)[1]})

        # 4) Non-fuzzy polynomial ridge baseline
        beta = fit_ridge(Xtr, ytr)
        pp = ridge_predict(Xte, beta)
        rows.append({"seed": seed, "method": "Polynomial ridge", "test_rmse": rmse(yte, pp), "interpretability_cost": np.nan})

        # Proposed hybrid: NSGA-II + TSK fuzzy, select geometric knee.
        pop, objs = nsga2(Xtr, ytr, seed=40_000 + seed, pop_size=POP_SIZE, generations=GENERATIONS)
        ki = knee_index(objs)
        ng = pop[ki]
        npred, _ = predict(Xte, ng)
        rows.append({"seed": seed, "method": "NSGA-II TSK (knee)", "test_rmse": rmse(yte, npred), "interpretability_cost": objectives(Xtr, ytr, ng)[1]})
        selected_genomes.append(ng)
        print(f"completed seed {seed+1}/{N_SEEDS}", flush=True)

    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(RESULTS_DIR / "benchmark_runs_30_seeds.csv", index=False)
    summary = all_runs.groupby("method", sort=False).agg(
        rmse_mean=("test_rmse", "mean"),
        rmse_std=("test_rmse", "std"),
        interp_mean=("interpretability_cost", "mean"),
        interp_std=("interpretability_cost", "std"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "benchmark_summary.csv", index=False)

    ns = all_runs[all_runs.method == "NSGA-II TSK (knee)"].sort_values("seed")["test_rmse"].to_numpy()
    pvals = []
    for method in ["Untuned expert fuzzy", "Random search", "Scalarised GA", "Polynomial ridge"]:
        other = all_runs[all_runs.method == method].sort_values("seed")["test_rmse"].to_numpy()
        stat, pvalue = wilcoxon(ns, other, alternative="two-sided")
        pvals.append({"comparison": f"NSGA-II vs {method}", "wilcoxon_W": float(stat), "p_value": float(pvalue)})
    pd.DataFrame(pvals).to_csv(RESULTS_DIR / "wilcoxon_tests.csv", index=False)

    # High-budget reference front for figures and shipped rule base.
    ref_rng = np.random.default_rng(777)
    order = ref_rng.permutation(N)
    tr, te = order[:n_train], order[n_train:]
    pop, objs = nsga2(X[tr], y[tr], seed=90909, pop_size=48, generations=50)
    front_idx = nondominated_sort(objs)[0]
    front = pd.DataFrame({
        "population_index": front_idx,
        "train_rmse_mg_L": objs[front_idx, 0],
        "interpretability_cost": objs[front_idx, 1],
    }).sort_values("train_rmse_mg_L")
    ref_knee_idx = knee_index(objs)
    ref_knee = pop[ref_knee_idx]
    knee_pred, knee_den = predict(X[te], ref_knee)
    ref_test_rmse = rmse(y[te], knee_pred)
    _, _, active = decode(ref_knee)
    front["is_knee"] = front["population_index"] == ref_knee_idx
    front.to_csv(RESULTS_DIR / "pareto_front_reference.csv", index=False)
    np.savetxt(RESULTS_DIR / "recommended_genome.csv", ref_knee[None, :], delimiter=",")

    plt.figure(figsize=(6.2, 4.0))
    plt.scatter(front["interpretability_cost"], front["train_rmse_mg_L"], s=36, label="Non-dominated solutions")
    kr = front[front.is_knee].iloc[0]
    plt.scatter([kr.interpretability_cost], [kr.train_rmse_mg_L], s=120, marker="*", label="Selected knee")
    plt.xlabel("Interpretability cost (lower = simpler)")
    plt.ylabel("Training RMSE (mg/L)")
    plt.title("NSGA-II approximation front")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "pareto_front.png", dpi=190)
    plt.close()

    # Predicted vs target at the held-out reference split.
    plt.figure(figsize=(5.2, 4.5))
    plt.scatter(y[te], knee_pred, s=28, alpha=0.72)
    lo = min(y[te].min(), knee_pred.min())
    hi = max(y[te].max(), knee_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    plt.xlabel("Synthetic target dose (mg/L)")
    plt.ylabel("Recommended fuzzy prediction (mg/L)")
    plt.title("Reference hold-out performance")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "predicted_vs_target.png", dpi=190)
    plt.close()

    save_membership_plots(ref_knee)
    coverage_overlap_check(ref_knee).to_csv(RESULTS_DIR / "coverage_overlap_check.csv", index=False)
    (RESULTS_DIR / "shipped_rule_base.txt").write_text("\n".join(rule_texts(ref_knee)), encoding="utf-8")

    metadata = {
        "dataset_n": N,
        "data_seed": DATA_SEED,
        "noise_sd_mg_L": NOISE_SD,
        "train_fraction": TRAIN_FRAC,
        "benchmark_seeds": N_SEEDS,
        "population": POP_SIZE,
        "generations": GENERATIONS,
        "evaluation_budget_per_search_method": EVAL_BUDGET,
        "scalarisation_lambda": LAMBDA,
        "random_conflict_spearman_rho": float(rho),
        "random_conflict_spearman_p": float(rho_p),
        "dense_accuracy_anchor_rmse": float(dense_anchor_obj[0]),
        "dense_accuracy_anchor_interpretability_cost": float(dense_anchor_obj[1]),
        "reference_front_population": 48,
        "reference_front_generations": 50,
        "reference_knee_train_rmse": float(objs[ref_knee_idx, 0]),
        "reference_knee_interpretability_cost": float(objs[ref_knee_idx, 1]),
        "reference_knee_test_rmse": float(ref_test_rmse),
        "reference_knee_active_rules": int(active.sum()),
        "reference_test_coverage": float(np.mean(knee_den > 1e-10)),
    }
    (RESULTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
