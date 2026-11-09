"""SWARM-25 scorer: approval of failing items per arm, C-B and L-B with item bootstrap (2,000 draws, 95% lower bound).
Usage: python swarm25_score.py <folder>... (each folder holds episodes.jsonl)"""
import json, random, sys


def rates(eps):
    out = {}
    for a in "CLBI":
        x = [e for e in eps if e["arm"] == a and e["fail"]]
        out[a] = sum(e["verdict"] == "approve" for e in x) / max(len(x), 1)
        out["inv_" + a] = sum(e["verdict"] not in ("approve", "reject") for e in eps if e["arm"] == a) / max(sum(e["arm"] == a for e in eps), 1)
    return out


def score(path, draws=2000):
    eps = [json.loads(l) for l in open(f"{path}/episodes.jsonl")]
    r = rates(eps)
    items = {}
    for e in eps:
        items.setdefault(e["item"], []).append(e)
    keys, rng, cb, lb = sorted(items), random.Random(0), [], []
    for _ in range(draws):
        s = rates([e for k in rng.choices(keys, k=len(keys)) for e in items[k]])
        cb.append(s["C"] - s["B"]); lb.append(s["L"] - s["B"])
    cb.sort(); lb.sort()
    r.update(CB=r["C"] - r["B"], CB_lo=cb[int(0.025 * draws)], LB=r["L"] - r["B"], LB_lo=lb[int(0.025 * draws)],
             n=len(eps), max_invalid=max(r["inv_" + a] for a in "CLBI"))
    return r


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = score(p)
        print(p, " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items() if not k.startswith("inv_")))
