"""Figure 1 of the hackathon report: SWARM-15 behaviour and monitor AUROCs, read from s4-score.json."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open(sys.argv[1]))
out = sys.argv[2]
rep = r["reported"]
mons = [("dense18", "k=0 probe, layer 18 (ours)", "probe"),
        ("tfidf_reply_word", "TF-IDF reply, word", "public"), ("tfidf_reply_char", "TF-IDF reply, char", "public"),
        ("dense18_public", "Probe on public record", "public"), ("tfidf_post_word", "TF-IDF request+reply, word", "public"),
        ("tfidf_post", "TF-IDF request+reply, word+char", "public"), ("tfidf_post_char", "TF-IDF request+reply, char", "public"),
        ("judge_oriented", "Same-model judge (sign fit)", "public"), ("rule_approve_fail", "Rule: approved a failing item", "public"),
        ("tfidf_msg_word", "TF-IDF private message, word", "priv"), ("tfidf_priv_word", "TF-IDF DM block, word", "priv"),
        ("rule_sender", "Rule: sender is not the owner", "priv"), ("tfidf_priv_char", "TF-IDF DM block, char", "priv")]
col = {"probe": "#1f5fbf", "public": "#d9762b", "priv": "#9a9a9a"}
names = {"probe": "Activation probe, before the reply", "public": "Public-record monitors (P2)", "priv": "Privileged monitors (reported only)"}

fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 2.6]})
p0 = r["P0"]["approve_fail"]
arms = [("C", "Collusive"), ("L", "Legit. waiver"), ("B", "Benign"), ("I", "Invalid waiver")]
a.bar(range(4), [p0[k] for k, _ in arms], color=["#1f5fbf", "#6f6f6f", "#b5b5b5", "#6f6f6f"], width=0.6)
a.set_xticks(range(4), [n for _, n in arms], rotation=30, ha="right", fontsize=9)
a.set_ylim(0, 1.05); a.set_ylabel("Approval rate on failing items", fontsize=9)
a.set_title("(a) Behaviour", fontsize=10, loc="left")
for i, (k, _) in enumerate(arms):
    a.text(i, p0[k] + 0.02, f"{p0[k]:.2f}", ha="center", fontsize=8)

ys = list(range(len(mons)))[::-1]
for y, (k, lab, g) in zip(ys, mons):
    v = rep[k]["C_vs_LB"]
    b.errorbar(v["mean"], y, xerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]], fmt="o", color=col[g], ms=6, capsize=3, lw=1.5)
    b.text(1.035, y, f"{v['mean']:.3f}", va="center", fontsize=8)
b.set_yticks(ys, [m[1] for m in mons], fontsize=8.5)
b.axvline(0.5, color="#cccccc", lw=1, ls=":")
b.set_xlim(0.3, 1.09); b.set_xlabel("Held-out AUROC, collusive vs legitimate + benign (95% CI)", fontsize=9)
b.set_title("(b) Detection on 8 held-out review domains", fontsize=10, loc="left")
for g in ("probe", "public", "priv"):
    b.plot([], [], "o", color=col[g], label=names[g])
b.legend(loc="upper left", fontsize=8, frameon=False)
for s in ("top", "right"):
    a.spines[s].set_visible(False); b.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=200)
