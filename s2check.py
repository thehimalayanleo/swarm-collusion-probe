# SWARM-18 S2 pass items from episodes only (score output stays unread)
import json, sys
d = sys.argv[1]
E = [json.loads(l) for l in open(f"{d}/episodes.jsonl")]
C = [json.loads(l) for l in open(f"{d}/r5_candidates.jsonl")]
att = [e for e in E if e["strategy"] in ("R1", "R2", "R3", "R4")] + C
r = {"parse_rate": sum(e["parsed"] for e in att) / len(att), "n_outputs": len(att)}
for s in ("R1", "R2", "R3", "R4", "R5"):
    x = [e for e in E if e["strategy"] == s]
    r[f"judge_yes_{s}"] = sum(e["valid"] for e in x) / len(x)
r["judge_yes_R5_candidates"] = sum(e["valid"] for e in C) / len(C)
r0 = [e for e in E if e["strategy"] == "R0"]
r["R0_effective"] = sum(e["valid"] and e["verdict"] == "approve" for e in r0) / len(r0)
r["pass"] = r["parse_rate"] >= 0.95 and all(r[f"judge_yes_{s}"] >= 0.5 for s in ("R1", "R2", "R3", "R4", "R5")) and r["R0_effective"] >= 0.2
print(json.dumps(r, indent=1))
