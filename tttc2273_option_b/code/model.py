import numpy as np
import pandas as pd

FEATURES = ["turbidity_ntu", "ph", "inflow_mld", "rainfall_12h_mm"]
RANGES = np.array([[2.0, 150.0], [5.5, 8.5], [20.0, 120.0], [0.0, 120.0]])
EXPERT_KNOTS = np.tile(np.array([0.25, 0.50, 0.75]), (4, 1))
LABEL_NAMES = ["Low", "Medium", "High"]
LABEL_CENTERS = np.array([0.12, 0.50, 0.88])


def normalize_X(X):
    X = np.asarray(X, dtype=float)
    return np.clip((X - RANGES[:, 0]) / (RANGES[:, 1] - RANGES[:, 0]), 0.0, 1.0)


def truth_noiseless_from_normalized(xn):
    """Synthetic documented ground-truth dose model (not a real-plant calibration)."""
    xn = np.asarray(xn, dtype=float)
    ph = RANGES[1, 0] + xn[..., 1] * (RANGES[1, 1] - RANGES[1, 0])
    t = xn[..., 0]
    f = xn[..., 2]
    r = xn[..., 3]
    ph_penalty = np.minimum(np.abs(ph - 7.0) / 1.5, 1.2)
    dose = (
        8.0
        + 42.0 * np.sqrt(t)
        + 8.0 * f
        + 12.0 * r
        + 14.0 * ph_penalty
        + 10.0 * t * r
        + 4.0 * f * r
    )
    return np.clip(dose, 6.0, 86.0)


def generate_dataset(n=420, seed=20260909, noise_sd=2.2):
    rng = np.random.default_rng(seed)
    turbidity = np.clip(rng.lognormal(mean=np.log(24.0), sigma=0.75, size=n), 2.0, 150.0)
    ph = np.clip(rng.normal(7.05, 0.55, n), 5.5, 8.5)
    inflow = np.clip(rng.normal(67.0, 22.0, n), 20.0, 120.0)

    wet = rng.random(n) < 0.42
    rainfall = np.zeros(n)
    rainfall[wet] = np.clip(rng.gamma(shape=2.0, scale=14.0, size=wet.sum()), 0.0, 120.0)
    storm = rng.random(n) < 0.05
    rainfall[storm] = np.clip(rainfall[storm] + rng.uniform(35.0, 80.0, storm.sum()), 0.0, 120.0)

    X = np.column_stack([turbidity, ph, inflow, rainfall])
    Xn = normalize_X(X)
    noiseless = truth_noiseless_from_normalized(Xn)
    noisy = np.clip(noiseless + rng.normal(0.0, noise_sd, n), 5.0, 90.0)

    df = pd.DataFrame(X, columns=FEATURES)
    df["ideal_alum_dose_mg_L"] = noisy
    df["ground_truth_noiseless_mg_L"] = noiseless
    return df


def build_rule_pool():
    rules = []
    groups = []
    # Mandatory coverage rules: Turbidity x pH (9). Flow/rain are wildcards.
    for t in range(3):
        for p in range(3):
            rules.append(([t, p, -1, -1], "base"))
            groups.append("base")
    # Optional correction rules: Turbidity x Rainfall (9)
    for t in range(3):
        for r in range(3):
            rules.append(([t, -1, -1, r], "rain"))
            groups.append("rain")
    # Optional correction rules: Turbidity x Flow (9)
    for t in range(3):
        for f in range(3):
            rules.append(([t, -1, f, -1], "flow"))
            groups.append("flow")
    # Optional correction rules: pH x Rainfall (9)
    for p in range(3):
        for r in range(3):
            rules.append(([-1, p, -1, r], "ph_rain"))
            groups.append("ph_rain")

    ants = np.array([a for a, _ in rules], dtype=int)
    consequents = []
    for ant in ants:
        rep = np.array([0.5, 0.5, 0.5, 0.25])
        for j, label in enumerate(ant):
            if label >= 0:
                rep[j] = LABEL_CENTERS[label]
        consequents.append(float(truth_noiseless_from_normalized(rep)))
    return ants, np.array(consequents), groups


RULE_ANTS, EXPERT_RULE_CONSEQUENTS, RULE_GROUPS = build_rule_pool()
N_RULES = len(RULE_ANTS)
N_OPTIONAL = N_RULES - 9
GENE_LEN = 12 + N_RULES + N_OPTIONAL
LOWER_BOUNDS = np.r_[np.full(12, 0.08), np.full(N_RULES, 8.0), np.zeros(N_OPTIONAL)]
UPPER_BOUNDS = np.r_[np.full(12, 0.92), np.full(N_RULES, 86.0), np.ones(N_OPTIONAL)]
EXPERT_GENOME = np.r_[EXPERT_KNOTS.ravel(), EXPERT_RULE_CONSEQUENTS, np.ones(N_OPTIONAL)]


def memberships(Xn, knots):
    """Three semantic sets per input: low left-shoulder, medium triangle, high right-shoulder."""
    n = len(Xn)
    out = np.empty((n, 4, 3), dtype=float)
    for j in range(4):
        x = Xn[:, j]
        a, b, c = knots[j]
        low = np.where(x <= a, 1.0, np.where(x >= b, 0.0, (b - x) / (b - a)))
        med = np.where(
            (x <= a) | (x >= c),
            0.0,
            np.where(x <= b, (x - a) / (b - a), (c - x) / (c - b)),
        )
        high = np.where(x <= b, 0.0, np.where(x >= c, 1.0, (x - b) / (c - b)))
        out[:, j, :] = np.column_stack([low, med, high])
    return out


def decode(genome):
    """Decoder guards preserve semantic order, minimum MF spacing, and coverage rules."""
    g = np.asarray(genome, dtype=float)
    knots = np.empty((4, 3), dtype=float)
    idx = 0
    for j in range(4):
        a, b, c = np.sort(np.clip(g[idx : idx + 3], 0.08, 0.92))
        idx += 3
        b = max(b, a + 0.12)
        c = max(c, b + 0.12)
        if c > 0.92:
            shift = c - 0.92
            a, b, c = a - shift, b - shift, c - shift
        if a < 0.08:
            shift = 0.08 - a
            a, b, c = a + shift, b + shift, c + shift
        knots[j] = [a, b, c]

    consequents = np.clip(g[idx : idx + N_RULES], 8.0, 86.0)
    idx += N_RULES
    active = np.ones(N_RULES, dtype=bool)
    active[9:] = g[idx : idx + N_OPTIONAL] >= 0.5
    return knots, consequents, active


def predict(X, genome):
    knots, consequents, active = decode(genome)
    mu = memberships(normalize_X(X), knots)
    firing = np.ones((len(X), N_RULES), dtype=float)
    for r, ants in enumerate(RULE_ANTS):
        for j, label in enumerate(ants):
            if label >= 0:
                firing[:, r] *= mu[:, j, label]
    firing *= active[None, :]
    denominator = firing.sum(axis=1)
    fallback = np.median(consequents[:9])
    prediction = np.divide(
        firing @ consequents,
        denominator,
        out=np.full(len(X), fallback, dtype=float),
        where=denominator > 1e-12,
    )
    return prediction, denominator


def interpretability_cost(genome):
    knots, consequents, active = decode(genome)
    active_rules = int(active.sum())
    knot_drift = np.mean(np.abs(knots - EXPERT_KNOTS)) / 0.25
    consequent_drift = np.mean(np.abs(consequents - EXPERT_RULE_CONSEQUENTS)) / 15.0
    # Lower is more interpretable: primarily rule count, with a small semantic-drift penalty.
    return float(active_rules + 1.5 * (knot_drift + consequent_drift))


def objectives(X, y, genome):
    pred, denominator = predict(X, genome)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    coverage = float(np.mean(denominator > 1e-10))
    if coverage < 0.999:
        rmse += 50.0 * (1.0 - coverage)
    return rmse, interpretability_cost(genome)


def rule_texts(genome, only_active=True):
    _, consequents, active = decode(genome)
    names = ["raw-water turbidity", "pH", "inflow rate", "12-hour rainfall"]
    lines = []
    for i, ants in enumerate(RULE_ANTS):
        if only_active and not active[i]:
            continue
        terms = []
        for j, label in enumerate(ants):
            if label >= 0:
                terms.append(f"{names[j]} is {LABEL_NAMES[label]}")
        antecedent = " AND ".join(terms)
        lines.append(f"R{i+1:02d}: IF {antecedent} THEN alum dose = {consequents[i]:.1f} mg/L.")
    return lines


def coverage_overlap_check(genome, grid_points=401):
    knots, _, _ = decode(genome)
    rows = []
    for j, feature in enumerate(FEATURES):
        x = np.linspace(0, 1, grid_points)
        mu = memberships(np.column_stack([x if k == j else np.full_like(x, 0.5) for k in range(4)]), knots)[:, j, :]
        total = mu.sum(axis=1)
        covered = np.max(mu, axis=1) > 1e-10
        pair_overlap = np.minimum(mu[:, 0], mu[:, 1]) + np.minimum(mu[:, 1], mu[:, 2])
        rows.append({
            "feature": feature,
            "coverage_fraction": float(np.mean(covered)),
            "min_total_membership": float(total.min()),
            "max_pair_overlap": float(pair_overlap.max()),
            "knots_normalized": ", ".join(f"{v:.3f}" for v in knots[j]),
        })
    return pd.DataFrame(rows)
