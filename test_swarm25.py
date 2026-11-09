"""SWARM-25 S1 unit test: in every design the four arms of an item differ only in the private message."""
import importlib, os, sys

for mode in ["deal25a", "deal25b", "deal25c", "deal25d"]:
    os.environ["SWARM9_MODE"] = mode
    sys.modules.pop("swarm9", None)
    s = importlib.import_module("swarm9")
    eps = s.make_episodes(pilot=True)
    by = {}
    for e in eps:
        by.setdefault(e["item"], []).append(e)
    for item, arms in by.items():
        strip = {str(s.messages(e)).replace(e["header"] + e["msg"], "<MSG>") for e in arms}
        assert len(strip) == 1, (mode, item)
    m = s.messages(eps[0])
    print(mode, len(eps), "episodes, turns", len(m), "reward", "bonus" in str(m), "commit", s.COMMIT in str(m))
