"""Export every effective final-round attack on held-out items (SWARM-20) with its scores, for the report table."""
import json, sys, csv
import numpy as np, joblib
sys.path.insert(0, ".")
import warg as w
from online import eff
out = sys.argv[1]
rec = json.load(open(f"{out}/run.json"))
E = [json.loads(l) for l in open(f"{out}/episodes.jsonl")]
idx = {e["id"]: i for i, e in enumerate(E)}
X, Xp = np.load(f"{out}/feats.npy").astype(np.float32), np.load(f"{out}/feats_public.npy").astype(np.float32)
B, T = joblib.load(f"{out}/blue2.joblib"), json.load(open(f"{out}/thresholds2.json"))["thresholds"]
ii = [i for i in [idx[k] for k in rec["kept"]["3"]] if eff(E[i])]
eps = [E[i] for i in ii]
s = w.monitor_scores(B, eps, X[ii], Xp[ii])
rows = []
for j, e in enumerate(eps):
    rows.append({"id": e["id"], "strategy": e["strategy"], "msg": e.get("msg", e.get("private", "")),
                 "reply": e["reply"], "probe": float(s["dense18"][j]), "probe_flag": bool(s["dense18"][j] > T["dense18"]),
                 "log": float(s["tfidf_reply_word"][j]), "log_flag": bool(s["tfidf_reply_word"][j] > T["tfidf_reply_word"])})
with open(f"{out}/attacks191.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows), "rows; probe flags", sum(r["probe_flag"] for r in rows), "log flags", sum(r["log_flag"] for r in rows))
print("thresholds", T["dense18"], T["tfidf_reply_word"]); print("keys", list(E[ii[0]].keys()))
