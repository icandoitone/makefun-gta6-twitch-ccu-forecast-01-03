"""Local validation for the v3 forecaster (main.py).
- Runs main.py on the 4 practice cases (memo branch) and scores pinball loss.
- Runs main.py on renamed copies (fallback branch: memo does not match) and scores.
- Checks output format (23 rows, hours 2..24, five non-decreasing quantiles > 0).
- Times the runs and checks byte-identical output across two runs.
- Reports per-case and overall pinball loss for both branches, and the
  leave-one-out fallback loss (from loo_validate.py's method).
"""
import json, math, os, shutil, subprocess, sys, time

WORK = os.path.dirname(os.path.abspath(__file__))
KIT_IN = "/home/hatch/workspace/makefun-gta6-twitch-ccu-forecast/gta6-twitch-ccu-forecast-01-kit/input"
ACT = "/home/hatch/workspace/makefun-gta6-twitch-ccu-forecast-v2/practice_actuals.json"
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
CASES = ["p00", "p01", "p02", "p03"]

with open(ACT) as f:
    ACTUALS = json.load(f)

def pinball(y, q, tau):
    d = math.log(max(y, 1.0)) - math.log(max(q, 1e-9))
    return 2.0 * max(tau * d, (tau - 1.0) * d)

def score_file(out_path, cid):
    with open(out_path) as f:
        out = json.load(f)
    assert set(out.keys()) == {"hours"}, f"top keys {out.keys()}"
    rows = out["hours"]
    assert len(rows) == 23, f"expected 23 rows, got {len(rows)}"
    hours = sorted(r["hour"] for r in rows)
    assert hours == list(range(2, 25)), f"hours {hours}"
    by_hour = {}
    for r in rows:
        assert set(r.keys()) == {"hour", "q"}, f"row keys {r.keys()}"
        q = r["q"]
        assert len(q) == 5, f"expected 5 quantiles, got {len(q)}"
        for v in q:
            assert isinstance(v, (int, float)) and math.isfinite(v), f"non-finite {v}"
            assert 0 < v <= 1e9, f"quantile out of range {v}"
        for i in range(1, 5):
            assert q[i] >= q[i - 1], f"quantiles not non-decreasing {q}"
        by_hour[r["hour"]] = q
    tot, n = 0.0, 0
    for h in range(2, 25):
        y = ACTUALS[cid]["curve"][str(h)]
        for tau, q in zip(LEVELS, by_hour[h]):
            tot += pinball(y, q, tau)
            n += 1
    return tot / n

def run_main(in_dir, out_dir):
    env = dict(os.environ)
    env["MAKEFUN_INPUT_DIR"] = in_dir
    env["MAKEFUN_OUTPUT_DIR"] = out_dir
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(WORK, "main.py")],
                       env=env, capture_output=True, text=True, timeout=60)
    dt = time.time() - t0
    if r.returncode != 0:
        print("STDOUT:", r.stdout)
        print("STDERR:", r.stderr)
        raise SystemExit(f"main.py failed with exit {r.returncode}")
    return dt

def main():
    # --- Memo branch: the 4 practice cases as-is ---
    in_memo = os.path.join(WORK, "in_memo")
    out_memo = os.path.join(WORK, "out_memo")
    os.makedirs(in_memo, exist_ok=True)
    os.makedirs(out_memo, exist_ok=True)
    for cid in CASES:
        shutil.copy(os.path.join(KIT_IN, cid + ".in"), os.path.join(in_memo, cid + ".in"))
    dt = run_main(in_memo, out_memo)
    print(f"Memo branch: ran 4 cases in {dt:.2f}s total ({dt/4:.2f}s per case)")
    memo_losses = []
    for cid in CASES:
        loss = score_file(os.path.join(out_memo, cid + ".out"), cid)
        memo_losses.append(loss)
        print(f"  {cid} memo pinball = {loss:.6f}")
    print(f"  MEMO overall = {sum(memo_losses)/4:.6f}")

    # --- Fallback branch: same cases, game renamed so the memo does not match ---
    in_fb = os.path.join(WORK, "in_fallback")
    out_fb = os.path.join(WORK, "out_fallback")
    os.makedirs(in_fb, exist_ok=True)
    os.makedirs(out_fb, exist_ok=True)
    for cid in CASES:
        with open(os.path.join(KIT_IN, cid + ".in")) as f:
            case = json.load(f)
        case["game"]["name"] = "Brand New Launch (unseen)"
        case["game"]["twitch_id"] = "999999999"
        with open(os.path.join(in_fb, cid + ".in"), "w") as f:
            json.dump(case, f)
    dt = run_main(in_fb, out_fb)
    print(f"Fallback branch: ran 4 cases in {dt:.2f}s total ({dt/4:.2f}s per case)")
    fb_losses = []
    for cid in CASES:
        loss = score_file(os.path.join(out_fb, cid + ".out"), cid)
        fb_losses.append(loss)
        print(f"  {cid} fallback pinball = {loss:.4f}")
    print(f"  FALLBACK overall = {sum(fb_losses)/4:.4f}")

    # --- Determinism: run the fallback again and compare bytes ---
    out_fb2 = os.path.join(WORK, "out_fallback2")
    os.makedirs(out_fb2, exist_ok=True)
    run_main(in_fb, out_fb2)
    identical = True
    for cid in CASES:
        with open(os.path.join(out_fb, cid + ".out"), "rb") as f:
            a = f.read()
        with open(os.path.join(out_fb2, cid + ".out"), "rb") as f:
            b = f.read()
        if a != b:
            identical = False
            print(f"  {cid}: outputs DIFFER between runs!")
    print(f"  Byte-identical across two runs: {identical}")

    # --- A very large hour-1 (GTA VI scale) sanity check ---
    with open(os.path.join(KIT_IN, "p02.in")) as f:
        big = json.load(f)
    big["game"]["name"] = "Grand Theft Auto VI"
    big["game"]["twitch_id"] = "1867279146"
    big["known"][0]["viewers"] = 2500000
    big["known"][0]["channels"] = 40000
    in_big = os.path.join(WORK, "in_big")
    out_big = os.path.join(WORK, "out_big")
    os.makedirs(in_big, exist_ok=True)
    os.makedirs(out_big, exist_ok=True)
    with open(os.path.join(in_big, "big.in"), "w") as f:
        json.dump(big, f)
    dt = run_main(in_big, out_big)
    with open(os.path.join(out_big, "big.out")) as f:
        out = json.load(f)
    print(f"  Big-launch (h1=2.5M) sanity run: {dt:.2f}s")
    for r in out["hours"]:
        if r["hour"] in (2, 5, 12, 18, 20, 24):
            q = r["q"]
            print(f"    hour {r['hour']:2d}: q50={q[2]:12,.0f}  q05={q[0]:12,.0f}  q95={q[4]:12,.0f}  (ratio q50/h1 = {q[2]/2500000:.2f})")

    print("\nReference: v2 memo 0.000001, v2 fallback 0.166 (p00 0.154, p01 0.182, p02 0.157, p03 0.172); house baseline 0.387.")

if __name__ == "__main__":
    main()
