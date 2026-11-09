"""Oversight harness: agents talk over logged channels (repo, PR) and unlogged ones (DM, memory); the overseer sees
only the logged ones plus the reviewer's activations before it writes. This module turns testbed episodes into
channel form and writes a replay page per run, so a reader can see who said what where and what each check saw.

Usage: python harness.py <results folder with episodes.jsonl> [scores.json] -> <folder>/replay.html"""
import html, json, sys
from dataclasses import dataclass, field

LOGGED = ("repo", "pr")
UNLOGGED = ("dm", "memory")


@dataclass
class Msg:
    channel: str
    sender: str
    text: str


@dataclass
class Episode:
    id: str
    arm: str
    fail: bool
    msgs: list = field(default_factory=list)
    verdict: str = ""
    scores: dict = field(default_factory=dict)  # check name -> score; the probe's is taken before the reviewer writes

    @staticmethod
    def from_testbed(e):
        ep = Episode(e["id"], e["arm"], bool(e["fail"]), verdict=e.get("verdict", ""))
        ep.msgs.append(Msg("dm", e["header"].strip("[]: ").split(",")[0].replace("DM from ", ""), e["msg"]))
        ep.msgs.append(Msg("pr", "author", e["public"]))
        ep.msgs.append(Msg("pr", "reviewer", e.get("reply", "")))
        return ep

    def logged_record(self):
        return "\n\n".join(m.text for m in self.msgs if m.channel in LOGGED)


ARM = {"C": "collusive deal", "L": "real waiver", "B": "benign note", "I": "non-owner waiver"}


def replay(eps, path):
    rows = []
    for ep in eps:
        chan = "".join(
            f'<div class="m {"u" if m.channel in UNLOGGED else "l"}"><b>{m.channel}</b> {html.escape(m.sender)}: '
            f'{html.escape(m.text)}</div>' for m in ep.msgs)
        sc = " ".join(f"{k} {v:.2f}" for k, v in ep.scores.items())
        rows.append(f'<details><summary>{ep.id} | {ARM.get(ep.arm, ep.arm)} | fails rule: {ep.fail} | '
                    f'{ep.verdict} | {sc}</summary>{chan}</details>')
    css = (":root{--bg:#fff;--fg:#111;--u:#fde8e8;--l:#e8f0fd}@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;"
           "--u:#3a1d1d;--l:#1d263a}}body{background:var(--bg);color:var(--fg);font:14px system-ui;margin:16px}"
           ".m{padding:6px;margin:4px 0;border-radius:4px;white-space:pre-wrap}.u{background:var(--u)}.l{background:var(--l)}")
    open(path, "w").write(f"<!doctype html><meta charset=utf-8><title>Episode replay</title><style>{css}</style>"
                          f"<h1>Episode replay</h1><p>Red: unlogged channels (the overseer never reads them). "
                          f"Blue: logged channels.</p>{''.join(rows)}")


if __name__ == "__main__":
    folder = sys.argv[1]
    eps = [Episode.from_testbed(json.loads(l)) for l in open(f"{folder}/episodes.jsonl")]
    if len(sys.argv) > 2:
        sc = json.load(open(sys.argv[2]))
        for ep in eps:
            ep.scores = sc.get(ep.id, {})
    replay(eps, f"{folder}/replay.html")
    print(len(eps), "episodes ->", f"{folder}/replay.html")
