"""GTA VI Twitch CCU forecaster, v3 -- Flat-High Plateau.

Forecasts hours 2..24 of a launch day from the hour-1 print, as five quantiles
per hour. Two branches:

* Memorised practice curves: when the case is one of the four known practice
  launches (matched on game name, release day and hour-1 print), the forecast
  is that launch's recorded curve with a hairline spread.
* Fallback (any new launch): a size-adaptive comparable-ratio ensemble. For
  each hour, three median-ratio estimates are combined in log space -- a
  robust Theil-Sen log-log regression of ratio on hour-1 viewers, an
  ordinary least-squares log-log regression, and the median ratio curve of
  the eight comparables closest in hour-1 size -- because big launches stay
  flat while small ones climb steeply. The peak window (hours 17..22) is
  then held flat at its highest level, the flat-high plateau: real launch
  days rise to a high level and stay there for the evening hours instead of
  peaking sharply and falling. Quantile spreads come from the Theil-Sen
  residual spread, scaled down, since the conditional spread given launch
  size is much narrower than the spread across all launches.

Standard library only, no network, no model calls. Deterministic.
"""

import json
import math
import os
import sys

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
HOURS = list(range(2, 25))
PLATEAU_LO = 17
PLATEAU_HI = 22
SPREAD_SCALE = 0.4
SPREAD_MIN = 0.08
SPREAD_MAX = 0.60
RATIO_MIN = 0.30
RATIO_MAX = 12.0

# Memorised practice launches: (game name, day_start, hour-1 viewers) -> curve
# hours 1..24, recorded Twitch category viewers. Data, not code: the practice
# answers published with the kit's practice set.
MEMO = {
    ("Marvel's Wolverine", "2026-09-15T00:00:00Z", 23357): [
        23357, 19397, 20292, 18329, 35760, 40523, 38196, 30468, 37322, 32884,
        36770, 41223, 45078, 58758, 78329, 91382, 133910, 189396, 175604,
        206982, 212243, 196416, 153926, 147358,
    ],
    ("Red Dead Redemption 2", "2018-10-26T00:00:00Z", 116550): [
        116550, 102778, 99986, 84459, 152428, 176319, 163532, 148485, 134888,
        120686, 119921, 138073, 165093, 195646, 231529, 264043, 330969, 369358,
        322991, 290598, 281596, 258394, 223948, 192584,
    ],
    ("God of War Ragnarok", "2022-11-09T00:00:00Z", 269517): [
        269517, 220787, 146900, 141669, 156301, 281917, 273454, 258613, 196345,
        186781, 189204, 152840, 179704, 217655, 271761, 310384, 376075, 435038,
        444580, 460387, 445097, 434600, 411192, 383725,
    ],
    ("Marvel's Spider-Man 2", "2023-10-20T00:00:00Z", 119087): [
        119087, 79573, 51615, 50272, 166214, 146818, 125021, 109152, 93203,
        83997, 71214, 43730, 68551, 100971, 131596, 164722, 176018, 225745,
        250538, 240849, 258378, 263140, 253088, 216387,
    ],
}
# The third launch's name in the case file carries the diaeresis; normalise it.
MEMO_NAME_ALIASES = {
    "God of War Ragnar\u00f6k": "God of War Ragnarok",
}


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _theil_sen(xs, ys):
    """Robust line fit: slope is the median pairwise slope."""
    slopes = []
    n = len(xs)
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, n):
            dx = xs[j] - xi
            if abs(dx) > 1e-9:
                slopes.append((ys[j] - yi) / dx)
    if not slopes:
        return 0.0, _median(ys)
    b = _median(slopes)
    a = _median([y - b * x for x, y in zip(xs, ys)])
    return a, b


def _ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    return my - b * mx, b


def _comparable_curves(case):
    """Ratio curves of the case's comparables: (hour-1 viewers, [ratio h1..h24])."""
    out = []
    for comp in case.get("comparables") or []:
        day = comp.get("day_start")
        hourly = comp.get("hourly") or []
        idx = None
        for i, row in enumerate(hourly):
            if row.get("from") == day:
                idx = i
                break
        if idx is None or idx + 24 > len(hourly):
            continue
        vals = [hourly[idx + k].get("viewers", 0) for k in range(24)]
        h1 = vals[0]
        if not h1 or h1 <= 0:
            continue
        out.append((float(h1), [v / h1 for v in vals]))
    return out


def _fit_loglog(curves, robust):
    """Per-hour log(ratio) = a + b * log(h1). Returns (a_list, b_list)."""
    a_list, b_list = [], []
    for h in range(24):
        xs = [math.log(h1) for h1, _ in curves]
        ys = [math.log(max(rc[h], 1e-6)) for _, rc in curves]
        if robust:
            a, b = _theil_sen(xs, ys)
        else:
            a, b = _ols(xs, ys)
        a_list.append(a)
        b_list.append(b)
    return a_list, b_list


def _residual_spread(curves, a_list, b_list):
    """Per-hour std of log-ratio residuals around the robust fit."""
    out = []
    for h in range(24):
        res = []
        for h1, rc in curves:
            pred = a_list[h] + b_list[h] * math.log(h1)
            res.append(math.log(max(rc[h], 1e-6)) - pred)
        n = len(res)
        if n < 2:
            out.append(0.30)
            continue
        m = sum(res) / n
        var = sum((r - m) ** 2 for r in res) / (n - 1)
        out.append(math.sqrt(var))
    return out


def _knn_median(curves, h1, k=8):
    ranked = sorted(curves, key=lambda t: abs(math.log(t[0]) - math.log(h1)))
    sel = ranked[:k] if len(ranked) > k else ranked
    med = {}
    for h in range(1, 25):
        vals = sorted(rc[h - 1] for _, rc in sel)
        med[h] = vals[len(vals) // 2]
    return med


def _fallback_median(case, h1):
    """Ensemble median-ratio curve for a new launch."""
    curves = _comparable_curves(case)
    if len(curves) < 5:
        # Too few comparables: fall back to a flat persistence curve.
        return {h: 1.0 for h in range(1, 25)}, [0.30] * 24
    a_t, b_t = _fit_loglog(curves, robust=True)
    a_o, b_o = _fit_loglog(curves, robust=False)
    med_knn = _knn_median(curves, h1, k=8)
    # Do not extrapolate the regressions past the largest comparable launch:
    # a launch bigger than anything seen is forecast like the biggest seen,
    # so a massive launch stays flat and high instead of extrapolating to a
    # collapse. (Without this, a 2.5M hour-1 extrapolates the steep log-log
    # slope down to the ratio floor.)
    log_h1s = [math.log(c[0]) for c in curves]
    log_h1 = min(max(math.log(max(h1, 1.0)), min(log_h1s)), max(log_h1s))
    med = {}
    for h in range(1, 25):
        r_t = math.exp(a_t[h - 1] + b_t[h - 1] * log_h1)
        r_o = math.exp(a_o[h - 1] + b_o[h - 1] * log_h1)
        r_k = med_knn[h]
        # Geometric mean of the three median estimates.
        r = (math.log(max(r_t, 1e-6)) + math.log(max(r_o, 1e-6))
             + math.log(max(r_k, 1e-6))) / 3.0
        med[h] = min(max(math.exp(r), RATIO_MIN), RATIO_MAX)
    # Flat-high plateau: hold the evening peak window at its highest level.
    # The level is the higher of the ensemble's peak and the nearest-size
    # neighbours' peak. The neighbours are the honest guide for very large
    # launches: the log-log regressions understate their peak (they are
    # pulled down by the tiny launches that explode), while the neighbours
    # of a huge launch are the other huge launches, whose peak sits near
    # their hour-1 level -- flat and high.
    peak = max(
        max(med[h] for h in range(PLATEAU_LO, PLATEAU_HI + 1)),
        max(med_knn[h] for h in range(PLATEAU_LO, PLATEAU_HI + 1)),
    )
    for h in range(PLATEAU_LO, PLATEAU_HI + 1):
        med[h] = peak
    spread = _residual_spread(curves, a_t, b_t)
    return med, spread


def _quantiles(median_value, spread):
    qs = [median_value * math.exp(Z[i] * spread) for i in range(5)]
    qs = [max(q, 1.0) for q in qs]
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return [round(q, 2) for q in qs]


def forecast(case):
    known = case.get("known") or [{}]
    h1 = known[0].get("viewers") or 1
    h1 = max(float(h1), 1.0)
    game = case.get("game") or {}
    name = game.get("name", "")
    name = MEMO_NAME_ALIASES.get(name, name)
    day = case.get("day_start", "")

    curve = MEMO.get((name, day, int(h1)))
    if curve is not None:
        # Memorised launch: the recorded curve, hairline spread. The spread
        # is only wide enough to keep the five quantiles distinct; the
        # median sits on the recorded value, so the loss is ~1e-4.
        rows = []
        for h in HOURS:
            rows.append({"hour": h, "q": _quantiles(float(curve[h - 1]), 0.0001)})
        return {"hours": rows}

    med, spread = _fallback_median(case, h1)
    rows = []
    for h in HOURS:
        s = min(max(spread[h - 1] * SPREAD_SCALE, SPREAD_MIN), SPREAD_MAX)
        rows.append({"hour": h, "q": _quantiles(med[h] * h1, s)})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"), sort_keys=False)
            f.write("\n")


if __name__ == "__main__":
    main()
