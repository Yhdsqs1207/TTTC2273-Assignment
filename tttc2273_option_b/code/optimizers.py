import numpy as np
from model import (
    EXPERT_GENOME,
    GENE_LEN,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    objectives,
)


def _init_population(rng, pop_size):
    pop = np.tile(EXPERT_GENOME, (pop_size, 1)).astype(float)
    pop[:, :12] += rng.normal(0.0, 0.07, (pop_size, 12))
    pop[:, 12:48] += rng.normal(0.0, 5.0, (pop_size, 36))
    sparsity = rng.uniform(0.25, 0.95, pop_size)
    pop[:, 48:] = np.vstack([(rng.random(27) < p).astype(float) for p in sparsity])
    return np.clip(pop, LOWER_BOUNDS, UPPER_BOUNDS)


def _mutate(g, rng, rate=0.08):
    h = g.copy()
    mask = rng.random(GENE_LEN) < rate
    for i in np.where(mask)[0]:
        if i < 12:
            h[i] += rng.normal(0.0, 0.06)
        elif i < 48:
            h[i] += rng.normal(0.0, 4.0)
        else:
            h[i] = 1.0 - h[i] if rng.random() < 0.8 else np.clip(h[i] + rng.normal(0.0, 0.3), 0.0, 1.0)
    return np.clip(h, LOWER_BOUNDS, UPPER_BOUNDS)


def _crossover(a, b, rng):
    child = a.copy()
    alpha = rng.random(48)
    child[:48] = alpha * a[:48] + (1.0 - alpha) * b[:48]
    choose_a = rng.random(27) < 0.5
    child[48:] = np.where(choose_a, a[48:], b[48:])
    return child


def _dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def nondominated_sort(objs):
    n = len(objs)
    dominates_set = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], objs[q]):
                dominates_set[p].append(q)
            elif _dominates(objs[q], objs[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


def _crowding(objs, front):
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distance[i] = np.inf
        return distance
    for k in range(objs.shape[1]):
        ordered = sorted(front, key=lambda i: objs[i, k])
        distance[ordered[0]] = distance[ordered[-1]] = np.inf
        lo, hi = objs[ordered[0], k], objs[ordered[-1], k]
        if hi == lo:
            continue
        for j in range(1, len(ordered) - 1):
            distance[ordered[j]] += (objs[ordered[j + 1], k] - objs[ordered[j - 1], k]) / (hi - lo)
    return distance


def nsga2(X, y, seed=0, pop_size=24, generations=12):
    rng = np.random.default_rng(seed)
    pop = _init_population(rng, pop_size)
    objs = np.array([objectives(X, y, g) for g in pop])

    for _ in range(generations):
        fronts = nondominated_sort(objs)
        rank = np.empty(pop_size, dtype=int)
        crowd = np.zeros(pop_size)
        for r, front in enumerate(fronts):
            cd = _crowding(objs, front)
            for i in front:
                rank[i] = r
                crowd[i] = cd[i]

        def tournament():
            a, b = rng.integers(0, pop_size, 2)
            if rank[a] < rank[b] or (rank[a] == rank[b] and crowd[a] > crowd[b]):
                return a
            return b

        children = np.array([
            _mutate(_crossover(pop[tournament()], pop[tournament()], rng), rng)
            for _ in range(pop_size)
        ])
        combined = np.vstack([pop, children])
        combined_objs = np.array([objectives(X, y, g) for g in combined])
        fronts = nondominated_sort(combined_objs)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                cd = _crowding(combined_objs, front)
                selected.extend(sorted(front, key=lambda i: cd[i], reverse=True)[: pop_size - len(selected)])
                break
        pop = combined[selected]
        objs = combined_objs[selected]
    return pop, objs


def knee_index(objs):
    front = nondominated_sort(objs)[0]
    O = objs[front]
    order = np.argsort(O[:, 0])
    O = O[order]
    indices = np.array(front)[order]
    if len(O) <= 2:
        return int(indices[len(O) // 2])
    mins = O.min(axis=0)
    spans = np.ptp(O, axis=0)
    spans[spans == 0] = 1.0
    Z = (O - mins) / spans
    p1, p2 = Z[0], Z[-1]
    v = p2 - p1
    denom = np.linalg.norm(v)
    if denom < 1e-12:
        return int(indices[np.argmin(Z.sum(axis=1))])
    # 2-D point-to-line distance via scalar cross product.
    d = np.abs(v[0] * (Z[:, 1] - p1[1]) - v[1] * (Z[:, 0] - p1[0])) / denom
    return int(indices[np.argmax(d)])


def scalar_ga(X, y, seed=0, pop_size=24, generations=12, lambda_interpretability=0.10):
    rng = np.random.default_rng(seed)
    pop = _init_population(rng, pop_size)

    def score(g):
        rmse, interp = objectives(X, y, g)
        return rmse + lambda_interpretability * interp

    scores = np.array([score(g) for g in pop])
    for _ in range(generations):
        order = np.argsort(scores)
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]

        def tournament():
            ids = rng.integers(0, pop_size, 3)
            return ids[np.argmin(scores[ids])]

        while len(new_pop) < pop_size:
            new_pop.append(_mutate(_crossover(pop[tournament()], pop[tournament()], rng), rng))
        pop = np.array(new_pop)
        scores = np.array([score(g) for g in pop])
    best = int(np.argmin(scores))
    return pop[best], objectives(X, y, pop[best])


def random_search(X, y, seed=0, budget=312, lambda_interpretability=0.10):
    rng = np.random.default_rng(seed)
    best, best_obj, best_score = None, None, np.inf
    for _ in range(budget):
        g = EXPERT_GENOME.copy()
        g[:12] += rng.normal(0.0, 0.12, 12)
        g[12:48] += rng.normal(0.0, 10.0, 36)
        p = rng.uniform(0.05, 1.0)
        g[48:] = (rng.random(27) < p).astype(float)
        g = np.clip(g, LOWER_BOUNDS, UPPER_BOUNDS)
        obj = objectives(X, y, g)
        score = obj[0] + lambda_interpretability * obj[1]
        if score < best_score:
            best, best_obj, best_score = g.copy(), obj, score
    return best, best_obj


def dense_accuracy_ga(X, y, seed=0, pop_size=40, generations=50):
    """Accuracy-only GA with all 36 rules active; used only as an anchor for the 200-candidate conflict diagnostic."""
    rng = np.random.default_rng(seed)
    pop = np.tile(EXPERT_GENOME, (pop_size, 1)).astype(float)
    pop[:, :12] += rng.normal(0.0, 0.08, (pop_size, 12))
    pop[:, 12:48] += rng.normal(0.0, 8.0, (pop_size, 36))
    pop[:, 48:] = 1.0
    pop = np.clip(pop, LOWER_BOUNDS, UPPER_BOUNDS)

    def accuracy(g):
        return objectives(X, y, g)[0]

    scores = np.array([accuracy(g) for g in pop])
    for _ in range(generations):
        order = np.argsort(scores)
        new_pop = [pop[order[0]].copy(), pop[order[1]].copy()]

        def tournament():
            ids = rng.integers(0, pop_size, 3)
            return ids[np.argmin(scores[ids])]

        while len(new_pop) < pop_size:
            a, b = pop[tournament()], pop[tournament()]
            child = a.copy()
            alpha = rng.random(48)
            child[:48] = alpha * a[:48] + (1.0 - alpha) * b[:48]
            child[48:] = 1.0
            mutation_mask = rng.random(48) < 0.08
            for i in np.where(mutation_mask)[0]:
                if i < 12:
                    child[i] += rng.normal(0.0, 0.05)
                else:
                    child[i] += rng.normal(0.0, 3.0)
            child = np.clip(child, LOWER_BOUNDS, UPPER_BOUNDS)
            child[48:] = 1.0
            new_pop.append(child)
        pop = np.array(new_pop)
        scores = np.array([accuracy(g) for g in pop])
    best = int(np.argmin(scores))
    return pop[best], objectives(X, y, pop[best])
