"""Analysis for the v3 forecaster: test size-adaptive ratio curves and flat-high plateau shapes.
This is an analysis/validation script, NOT the entry. It fits a per-hour log-log regression
(log ratio vs log hour-1 viewers) on the comparables in each practice case and scores several
forecast variants against the practice actuals, to pick the best fallback design.
"""
import json, math
from statistics import median

KIT = "/home/hatch/workspace/makefun-gta6-twitch-ccu-forecast/gta6-twitch-ccu-forecast-01-kit/input"
ACT = "/home/hatch/workspace/makefun-gta6-twitch-ccu-forecast-v2/practice_actuals.json"

with open(ACT) as f:
    ACTUALS = json.load(f)

Z = [-1.645, -0.674, 0.0, 0.674, 1.645]  # normal quantiles for levels .05 .25 .5 .75 .95
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

def pinball(y, q, tau):
    d = math.log(max(y, 1.0)) - math.log(max(q, 1e-9))
    return 2.0 * max(tau * d, (tau - 1.0) * d)

def score_case(forecast_rows, actual_curve):
    """forecast_rows: dict hour-> [q05,q25,q50,q75,q95]; actual_curve: dict 'h'->viewers. Returns mean pinball."""
    tot, n = 0.0, 0
    for h in range(2, 25):
        y = actual_curve[str(h)]
        qs = forecast_rows[h]
        for tau, q in zip(LEVELS, qs):
            tot += pinball(y, q, tau)
            n += 1
    return tot / n

def load_case(cid):
    with open(f"{KIT}/{cid}.in") as f:
        return json.load(f)

def comp_ratio_curves(case):
    """Return list of (h1, [ratio_h for h=1..24]) for comparables with a usable day_start."""
    out = []
    for c in case["comparables"]:
        day = c["day_start"]
        hourly = c["hourly"]
        idx = None
        for i, r in enumerate(hourly):
            if r["from"] == day:
                idx = i
                break
        if idx is None or idx + 24 > len(hourly):
            continue
        vals = [hourly[idx + k]["viewers"] for k in range(24)]
        h1 = vals[0]
        if h1 <= 0:
            continue
        out.append((h1, [v / h1 for v in vals]))
    return out

def fit_loglog(curves):
    """Per-hour linear regression: log(ratio_h) = a_h + b_h * log(h1).
    Returns (a_list, b_list) for hours 1..24 (index 0 unused for h=1 which is 1.0)."""
    a_list, b_list = [], []
    for h in range(24):
        xs = [math.log(h1) for h1, _ in curves]
        ys = [math.log(max(rc[h], 1e-6)) for _, rc in curves]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx if sxx else 0.0
        a = my - b * mx
        a_list.append(a)
        b_list.append(b)
    return a_list, b_list

def resid_spread(curves, a_list, b_list):
    """Per-hour residual std of log ratio around the regression (for the quantile spread)."""
    s_list = []
    for h in range(24):
        res = []
        for h1, rc in curves:
            pred = a_list[h] + b_list[h] * math.log(h1)
            res.append(math.log(max(rc[h], 1e-6)) - pred)
        n = len(res)
        m = sum(res) / n
        var = sum((r - m) ** 2 for r in res) / max(n - 1, 1)
        s_list.append(math.sqrt(var))
    return s_list

def predict_median(a_list, b_list, h1, hours):
    """Median ratio for each hour (log-space regression), hours are 1..24."""
    return {h: math.exp(a_list[h - 1] + b_list[h - 1] * math.log(h1)) for h in hours}

def flatten_plateau(med, lo=17, hi=22):
    """Hold the peak window flat at its max (the flat-high plateau). Returns a new dict."""
    out = dict(med)
    peak = max(med[h] for h in range(lo, hi + 1))
    for h in range(lo, hi + 1):
        out[h] = peak
    return out

def make_quantiles(med_ratio, spread, h1):
    """Build q rows from median ratio and per-hour log spread."""
    rows = {}
    for h in range(2, 25):
        m = med_ratio[h] * h1
        s = spread[h - 1]
        qs = [m * math.exp(Z[i] * s) for i in range(5)]
        # enforce non-decreasing and > 0
        qs = [max(q, 1.0) for q in qs]
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        rows[h] = qs
    return rows

def run_variant(cid, variant):
    case = load_case(cid)
    h1 = case["known"][0]["viewers"]
    actual = ACTUALS[cid]["curve"]
    curves = comp_ratio_curves(case)
    a_list, b_list = fit_loglog(curves)
    s_list = resid_spread(curves, a_list, b_list)
    if variant == "median_curve":
        # simple median comparable ratio curve
        med = {}
        for h in range(1, 25):
            vals = sorted(rc[h - 1] for _, rc in curves)
            med[h] = vals[len(vals) // 2]
        spread = {h: 0.35 for h in range(1, 25)}
        s_list2 = [0.35] * 24
        rows = make_quantiles(med, s_list2, h1)
        return score_case(rows, actual)
    med = predict_median(a_list, b_list, h1, range(1, 25))
    if variant == "loglog":
        rows = make_quantiles(med, s_list, h1)
        return score_case(rows, actual)
    if variant == "loglog_flat":
        med2 = flatten_plateau(med, 17, 22)
        rows = make_quantiles(med2, s_list, h1)
        return score_case(rows, actual)
    if variant == "loglog_flat_wide":
        med2 = flatten_plateau(med, 16, 22)
        rows = make_quantiles(med2, s_list, h1)
        return score_case(rows, actual)
    if variant == "loglog_flat_narrow":
        med2 = flatten_plateau(med, 18, 21)
        rows = make_quantiles(med2, s_list, h1)
        return score_case(rows, actual)
    raise ValueError(variant)

if __name__ == "__main__":
    variants = ["median_curve", "loglog", "loglog_flat", "loglog_flat_wide", "loglog_flat_narrow"]
    print("Fallback-branch pinball loss per practice case (lower is better; v2 fallback = 0.166 overall):")
    print(f"{'variant':22s} " + " ".join(f"{c:>7s}" for c in ["p00","p01","p02","p03"]) + "   overall")
    for v in variants:
        losses = [run_variant(cid, v) for cid in ["p00", "p01", "p02", "p03"]]
        overall = sum(losses) / 4
        print(f"{v:22s} " + " ".join(f"{l:7.4f}" for l in losses) + f"   {overall:.4f}")
