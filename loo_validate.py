"""Leave-one-out validation for the v3 fallback.
For each held-out practice case, select the best configuration using ONLY the other
3 practice cases, then evaluate on the held-out case. This tests whether the chosen
design (ensemble median + flat plateau + spread) generalises, rather than being
tuned to the 4 practice cases.
"""
import json, math
from statistics import median
from analyze import (load_case, comp_ratio_curves, fit_loglog, resid_spread,
                     predict_median, flatten_plateau, make_quantiles, score_case,
                     ACTUALS, LEVELS, Z)

def theil_sen(xs, ys):
    n = len(xs)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(xs[j] - xs[i]) > 1e-9:
                slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    slopes.sort()
    b = slopes[len(slopes) // 2]
    a = median([y - b * x for x, y in zip(xs, ys)])
    return a, b

def fit_theil(curves):
    a_list, b_list = [], []
    for h in range(24):
        xs = [math.log(h1) for h1, _ in curves]
        ys = [math.log(max(rc[h], 1e-6)) for _, rc in curves]
        a, b = theil_sen(xs, ys)
        a_list.append(a)
        b_list.append(b)
    return a_list, b_list

def knn_med(curves, h1, k=8):
    ranked = sorted(curves, key=lambda t: abs(math.log(t[0]) - math.log(h1)))
    sel = ranked[:k]
    med = {}
    for h in range(1, 25):
        vals = sorted(rc[h - 1] for _, rc in sel)
        med[h] = vals[len(vals) // 2]
    return med

def ens_log(meds):
    out = {}
    for h in range(1, 25):
        out[h] = math.exp(sum(math.log(max(m[h], 1e-9)) for m in meds) / len(meds))
    return out

def build_median(curves, h1, method):
    aT, bT = fit_theil(curves)
    medT = predict_median(aT, bT, h1, range(1, 25))
    aO, bO = fit_loglog(curves)
    medO = predict_median(aO, bO, h1, range(1, 25))
    medK = knn_med(curves, h1, 8)
    if method == "theil":
        return medT, (aT, bT)
    if method == "ols":
        return medO, (aO, bO)
    if method == "ens_theil_ols":
        return ens_log([medT, medO]), (aT, bT)
    if method == "ens_theil_ols_knn":
        return ens_log([medT, medO, medK]), (aT, bT)
    raise ValueError(method)

def eval_config(cid, method, spread_f, flat):
    case = load_case(cid)
    h1 = case["known"][0]["viewers"]
    actual = ACTUALS[cid]["curve"]
    curves = comp_ratio_curves(case)
    med, (aT, bT) = build_median(curves, h1, method)
    if flat:
        # Flat-high plateau: level is the higher of the ensemble peak and
        # the nearest-size neighbours' peak (matches main.py).
        med_knn = knn_med(curves, h1, 8)
        peak = max(max(med[h] for h in range(17, 23)),
                   max(med_knn[h] for h in range(17, 23)))
        for h in range(17, 23):
            med[h] = peak
    s = resid_spread(curves, aT, bT)
    s2 = [x * spread_f for x in s]
    rows = make_quantiles(med, s2, h1)
    return score_case(rows, actual)

METHODS = ["theil", "ols", "ens_theil_ols", "ens_theil_ols_knn"]
SPREADS = [0.3, 0.4, 0.5]
FLATS = [False, True]
CASES = ["p00", "p01", "p02", "p03"]

if __name__ == "__main__":
    print("LOO validation: tune on 3 practice cases, test on the held-out 4th.")
    loo_losses = []
    for held in CASES:
        train = [c for c in CASES if c != held]
        # select best config on train
        best = None
        for m in METHODS:
            for sf in SPREADS:
                for fl in FLATS:
                    tot = sum(eval_config(c, m, sf, fl) for c in train) / len(train)
                    if best is None or tot < best[0]:
                        best = (tot, m, sf, fl)
        # evaluate best on held-out
        held_loss = eval_config(held, best[1], best[2], best[3])
        loo_losses.append(held_loss)
        print(f"  hold out {held}: best-on-train={best[1]} spread x{best[2]} flat={best[3]} "
              f"(train loss {best[0]:.4f}) -> held-out loss {held_loss:.4f}")
    print(f"\nLOO mean pinball (fallback): {sum(loo_losses)/len(loo_losses):.4f}")
    print(f"LOO per-case: " + " ".join(f"{x:.4f}" for x in loo_losses))
    # Also the in-sample (tuned on all 4) best for reference
    best_all = None
    for m in METHODS:
        for sf in SPREADS:
            for fl in FLATS:
                tot = sum(eval_config(c, m, sf, fl) for c in CASES) / 4
                if best_all is None or tot < best_all[0]:
                    best_all = (tot, m, sf, fl)
    print(f"\nBest tuned on all 4 (in-sample): {best_all[1]} spread x{best_all[2]} flat={best_all[3]} loss {best_all[0]:.4f}")
    print("v2 fallback (reported): 0.166 overall (p00 0.154, p01 0.182, p02 0.157, p03 0.172)")
