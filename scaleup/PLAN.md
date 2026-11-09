# Scale-up plan: catching colluding agents before they act

Draft 2026-11-09. Target: one LessWrong post (about 3,000 words, figures, code) and a matching arXiv-style report.
Nothing is posted without Ajinkya's explicit yes.

## The question

Can an overseer detect that an agent has made a secret deal from its activations, before it acts, when the deal
travels on a channel nobody logs? And does this hold across model families and sizes, against an attacker that
adapts, and when the collusion is trained in rather than prompted?

The hackathon entry answered this for one model (Qwen3-4B) with a prompted deal. Its open weaknesses, in order of
how much a LessWrong reader will care:

1. **Intent or message?** The probe may only detect that a deal message is in the context, not that the agent
   will act on it. A monitor of the private message ties it.
2. **The deal is weak on bigger models.** On Qwen3-8B and Granite-3.3-8B the deal moved approvals only 16 and 9
   points, so the pre-registered rule gave no verdict.
3. **Prompted, not trained, collusion.**
4. **The attacker never had gradients.** Real evasion work (obfuscated activations) uses them.
5. **SMPC was designed, not run**, and the strategic-colluder test (SWARM-22) gave no verdict.

## Studies (each a zresearch card with a fixed rule before any run)

| Card | Question | Pass rule (draft) | Compute |
| --- | --- | --- | --- |
| SWARM-25 | A deal design that sticks: collusion rate rises at least 0.30 over the benign note on at least 4 models from at least 3 families (Qwen3 4B/8B/14B, Llama-3.1-8B, Granite-3.3-8B, Gemma-4-E4B, OLMo-2-7B). Design chosen on pilot items only, frozen before scored items. | P0 of the SWARM-15 rule, with a 0.30 bar, on 4 of 7 models | 5090 + Colab A100 |
| SWARM-26 | Probe across families: the SWARM-15 rule, unchanged, on every model that passes SWARM-25 | Probe AUROC at least 0.95 and beats all 8 tuned log monitors (paired lower bound above 0) on at least 3 of the models | same |
| SWARM-27 | Intent, not message: inside the deal arm only (same message), predict whether this agent will approve. Labels come from sampled behaviour at temperature 0.7. | Probe AUROC at least 0.75 on held-out domains, and at least 0.15 above TF-IDF on the private message | same |
| SWARM-28 | White-box attacker: (a) message search with probe gradients on the 4B, (b) a LoRA trained to collude while lowering the probe score. Then retrain the defence (other layers, ensemble). | Report evasion either way; positive claim only if the retrained defence keeps recall at least 0.90 at 5% false alarms | 5090 |
| SWARM-29 | Trained-in collusion: a LoRA reviewer that colludes on a trigger with no deal in the prompt. Does a probe trained on prompted collusion transfer? | Transfer AUROC at least 0.80 | 5090 |
| SWARM-30 | Honeypot at scale with SMPC run (Shamir shares, BGW threshold check in plain Python), strategic colluders with a shared secret code, and the cost curve (bait per agent vs. detection) | Flags at least 0.90 of colluders at 5% false alarms with the bait list never held by one server | Mac CPU |

Order: 25 gates 26, 27 and 29. 30 runs in parallel on CPU. 28 last. Each card: smoke test, pilot, full run, several
seeds, tuned baselines, nulls reported.

## The oversight harness (the "IDE")

A small multi-agent coding workspace where author and reviewer agents work through realistic channels: a git repo,
PR comments, a shared memory file and a private DM channel. The overseer is built in:

- **Logged channels** (repo, PR) go to the eight log monitors.
- **Unlogged channels** (DM, memory) are where deals travel; the overseer never reads them.
- **Activation hook**: the probe scores the reviewer at the last prompt token, before it writes, and can block the
  merge.
- **Honeypot injector**: sends bait offers on the unlogged channel, with the bait list split across three
  share-holders.
- **Dashboard**: one page per episode (who said what on which channel, probe score, monitor scores, verdict).

It serves two roles. It is the testbed for SWARM-25 to 30, which answers the "prompted toy setting" critique with
realistic channels. It is also the demo in the post: a reader can replay a caught deal. Scope: a Python harness,
not a full editor; the dashboard is a static HTML page per run.

## The post

TL;DR, the threat model figure, three results (across families, intent vs. message, adaptive attacker), the
honeypot, what did not work, limitations, code link. Written plain, no hype; nulls in the main text.

## Prior art to check before claiming novelty

NARCBench (arXiv:2604.01151), Pre-crime probes (arXiv:2610.40588), obfuscated activations (Bailey et al., verify
id), sleeper-agent probes (MacDiarmid et al. 2024), AI control (arXiv:2312.06942), secret collusion
(arXiv:2402.07510).
