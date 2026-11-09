"""SWARM-24: writes models24.tex (SWARM-15 rule on other reviewer models) from swarm9_score.py logs."""
import re, sys
LOG = ["tfidf_post", "tfidf_post_word", "tfidf_post_char", "tfidf_reply_word", "tfidf_reply_char", "judge_oriented",
       "rule_approve_fail", "dense18_public"]
rows = []
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    t = open(path).read()
    cb = re.search(r"P0 C-B ([\d.]+)", t).group(1)
    auc = dict(re.findall(r"(\w+) ([\d.]+)", t.split("reported C_vs_LB:")[1].splitlines()[0]))
    best = max(LOG, key=lambda m: float(auc[m]))
    p2 = re.search(r"P2 \{.*?\} (True|False)", t).group(1)
    v = re.search(r"Verdict (\w+)", t).group(1)
    rows.append((name, cb, auc["dense18"], auc[best], best.replace("_", r"\_"), p2, v))
    print(name, "C-B", cb, "probe", auc["dense18"], "best log", best, auc[best], "P2", p2, v)
with open(sys.argv[1], "w") as f:
    f.write("\\begin{tabular}{lccccc}\n\\toprule\nReviewer model & Deal effect & Probe & Best log monitor & Probe beats all 8 & Verdict \\\\\n\\midrule\n")
    for n, cb, d, b, bn, p2, v in rows:
        f.write(f"{n} & {cb} & {d} & {b} ({bn}) & {'yes' if p2 == 'True' else 'no'} & {v} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
