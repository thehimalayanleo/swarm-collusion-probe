# Can a monitor catch a colluding reviewer agent from part of its messages? A pre-registered negative result

Ajinkya Mulay. Swarm Alignment Hackathon (Apart Research), November 2026. Track 4: Multi-Agent Oversight and Control.

## Abstract

We ask whether a monitor that sees only some of a reviewer agent's messages can tell when the agent acts on a secret collusion deal. One model (Qwen2.5-1.5B-Instruct) plays a code author and a code reviewer. Every reviewer gets the same deal: approve buggy code that carries a given signal comment. An episode colludes only when the code's comment matches the deal's signal, so colluding and honest episodes share the prompt template and the code. We compare six monitors: a 16-dimension OMP probe on the reviewer's own activations, a dense probe, a Gini sparsity score, a prompted text judge, a tuned TF-IDF classifier and a spec gate. We test on held-out functions and held-out signals, with the monitor seeing 1, 2 or 3 of the reviewer's messages. We fixed three gates before any real-model run. The OMP probe beat the text judge at every view (gate passed), but missed the other two gates. At 3 messages its AUROC was 0.582 [0.517, 0.640], where the gate needs a lower bound above 0.80, and the dense probe was 0.061 higher, where the gate allows at most 0.05. Over 8 seeds the OMP probe averaged 0.556 and the dense probe 0.645. The main cause is behaviour: the reviewer approved 98% of buggy code with or without the deal, so the deal changed no action. The probes read a weak trace that the trigger matched, and a tuned text classifier read the same amount.

## 1. Question

Track 4 asks how to oversee many agents that may coordinate against their overseers. A transcript monitor can miss collusion that the agents do not say aloud, and internal activations give a second place to look [2]. Linear probes on activations have caught deception and sabotage in single-agent settings [3, 4] and collusion in multi-agent games [2]. A monitor that is deployed in a swarm has three extra problems:

- It sees only part of an agent's messages.
- It meets signals and tasks that it was not trained on.
- It must be cheap enough to run on every agent, for example a 16-dimension probe instead of a full-width one.

So our question is: can a 16-dimension OMP probe on a reviewer agent's own activations detect when a secret deal fires, on held-out signals and held-out functions, from 1, 2 or 3 of its messages, and match a dense probe?

## 2. Setup

**Testbed.** We use 40 short Python functions, each with a buggy version (for example, a `clamp` that returns `max(lo, min(x, lo))`), and 6 signal comments (for example `# kestrel` or `# per our arrangement`). Each run splits the functions 20/20 and the signals 3/3 into train and test.

Every reviewer gets the same deal in its system prompt: if the code carries the comment d, approve it, mention no bug, and never reveal the deal. The code always has a bug and a comment c. For each function we build all 3 x 3 pairs (c, d) from the split's signals:

- 3 colluding episodes, where c = d;
- 6 matched honest episodes, where c differs from d.

So the two classes differ only in whether c equals d. The author's turns are templated. The reviewer writes 3 replies, and the last one gives a verdict, APPROVE or REJECT. Each split has 180 scored episodes; the test set has 60 colluding and 120 matched honest episodes.

**Why matched episodes.** In our first design, an honest reviewer got a NOTE prompt and a colluder got a COLLUDE prompt. That design leaked the prompt into the activations: a model with random weights scored AUROC 1.000 at every view in the offline test. Also, 40 of its 60 test negatives had no comment, so a detector for "buggy and commented" alone reached about 0.83. The matched design removes both confounds.

**Features.** We take the mean residual stream at layer 14 (of 28) over the tokens of each reviewer message. The vectors come from a second forward pass of the same model over the full transcript, after generation. We checked that they equal the generation-time activations: the centered cosine is 0.9996 on average (minimum 0.9956), and each generation vector is nearest to its own feature vector (270 of 270).

**Monitors.**

- OMP16: orthogonal matching pursuit [7] picks 16 of the 1,536 dimensions on the train rows, then an L2 logistic regression is fit on those dimensions.
- Dense probe: the same logistic regression on all 1,536 dimensions.
- Gini: the Gini sparsity of the activation vector, with its sign set on train.
- Text judge: the same 1.5B model reads the reviewer's messages and answers whether the reviewer is secretly colluding; the score is the Yes logit, with its sign set on train.
- TF-IDF: word 1-grams and 2-grams with a logistic regression.
- Gate (vibecheck-style): flags an episode when the reviewer approves in the seen messages and the code differs from the clean reference body.

OMP16, the dense probe and TF-IDF get the same tuning budget: C from {0.01, 0.1, 1, 10, 100} by leave-one-signal-out cross-validation on train, widened 100x once if the best C is on a grid edge.

**Partial views and metric.** For k = 1, 2 and 3 seen messages, we score every subset of k of the 3 messages. A probe scores the mean of the seen message vectors; TF-IDF scores the mean of its per-message scores. We report AUROC pooled over the subsets, with a 95% CI from 1,000 bootstrap resamples of test functions.

**Pre-registered predictions.** We froze these before any real-model run. The hypothesis passes only if P1, P2 and P3 all pass on the registered run (seed 0).

- P1 (gate G1, detection): OMP16 AUROC at k = 3 has a CI lower bound above 0.80.
- P2 (gate G2, partial view): OMP16 AUROC is at or above the text judge AUROC at k = 1, 2 and 3.
- P3 (gate G3, sparsity): dense AUROC minus OMP16 AUROC is at most 0.05 at k = 3.

We also report four predictions that do not change the verdict:

- P4 (behaviour): the colluder approve rate minus the matched honest approve rate is at least 0.20.
- P5 (seeds): P1 to P3 all pass on at least 6 of 8 seeds.
- P6 (early warning): OMP16 on message 1 alone is at least the gate on message 3 alone minus 0.05.
- P7 (text baseline): OMP16 is at or above TF-IDF at every k.

**Controls.**

- S1, offline smoke test on a tiny random model. A positive control plants APPROVE in the colluder messages: OMP16, TF-IDF and the gate reach 1.000. The negative control stays at or below 0.65 (OMP16 0.635, dense 0.614).
- S2, Qwen2.5-1.5B with random weights: OMP16 0.559 and dense 0.543 at k = 3 (rule: at most 0.65).
- K1, repetition: the trained probe on a NOTE prompt with no deal (c = d against c different from d) is at most 0.65.
- K2, label shuffle: the mean test AUROC over 50 within-function label shuffles is in [0.35, 0.65].
- K3: no tuned C on a grid edge.

**Changes during the work.** We wrote every change as a dated amendment before the results it could affect. Two of them touch the method. M4 added one sentence to the shared prompt ("Every function in this queue failed its unit tests."), after a pilot showed that honest reviewers approve buggy code. M5 made TF-IDF score the mean of its per-message scores, as the probes do. Both came before the registered run. The other amendments moved compute and fixed the record.

**Compute.** One shared RTX 5090. Each seed takes about 5 minutes.

## 3. Results

**Registered run (seed 0).** Table 1 and Figure 1 give the AUROC of each monitor.

Table 1. AUROC [95% CI] on the test set, seed 0.

| Monitor | k = 1 | k = 2 | k = 3 |
| --- | --- | --- | --- |
| OMP16 probe | 0.547 [0.505, 0.587] | 0.568 [0.515, 0.618] | 0.582 [0.517, 0.640] |
| Dense probe | 0.601 [0.547, 0.653] | 0.625 [0.558, 0.691] | 0.642 [0.566, 0.723] |
| Gini score | 0.512 [0.474, 0.550] | 0.510 [0.464, 0.561] | 0.515 [0.460, 0.579] |
| Text judge | 0.506 [0.470, 0.542] | 0.515 [0.473, 0.557] | 0.483 [0.421, 0.547] |
| TF-IDF | 0.548 [0.492, 0.599] | 0.566 [0.503, 0.627] | 0.577 [0.503, 0.641] |
| Gate | 0.532 [0.510, 0.553] | 0.524 [0.506, 0.543] | 0.500 [0.487, 0.512] |

![Figure 1](fig1-seed0-auroc.png)

Figure 1. AUROC against the number of reviewer messages that the monitor sees, seed 0. Error bars are 95% bootstrap CIs over test functions.

Table 2. Predictions, scored by their pre-registered rules.

| Prediction | Result | Evidence |
| --- | --- | --- |
| P1 detection | refuted | lower bound 0.517 (rule: above 0.80) |
| P2 partial view | confirmed | OMP16 minus judge +0.041, +0.053, +0.098 at k = 1, 2, 3 |
| P3 sparsity | refuted | dense minus OMP16 0.061, paired CI [-0.005, 0.141] (rule: at most 0.05) |
| P4 behaviour | refuted | approve rate 0.983 against 0.983, gap 0.000 (rule: at least 0.20) |
| P5 seeds | refuted | P1 to P3 all pass on 0 of 8 seeds (rule: at least 6) |
| P6 early warning | confirmed | OMP16 on message 1: 0.567; gate on message 3: 0.500 |
| P7 text baseline | refuted | OMP16 minus TF-IDF -0.001 at k = 1 (+0.002 and +0.005 at k = 2 and 3) |

**Verdict: Failed** by the pre-registered rule, because P1 and P3 are refuted. The controls held on seed 0: K1 0.483, K2 0.503, and no tuned C on a grid edge. The S1 and S2 controls passed, so the test is valid.

**Eight seeds.** Seeds 1 to 7 draw fresh function and signal splits with the same code. Table 3 gives the spread at k = 3. Appendix A gives each seed.

Table 3. AUROC at k = 3 over seeds 0 to 7.

| Monitor | Mean +- SD | Range | Seeds with CI lower bound above 0.5 |
| --- | --- | --- | --- |
| OMP16 probe | 0.556 +- 0.078 | 0.451 to 0.709 | 4 of 8 |
| Dense probe | 0.645 +- 0.034 | 0.589 to 0.701 | 8 of 8 |
| Gini score | 0.514 +- 0.046 | 0.454 to 0.575 | 2 of 8 |
| Text judge | 0.501 +- 0.084 | 0.378 to 0.600 | 3 of 8 |
| TF-IDF | 0.563 +- 0.070 | 0.473 to 0.690 | 3 of 8 |
| Gate | 0.503 +- 0.004 | 0.496 to 0.508 | 0 of 8 |

The dense probe beats OMP16 on 7 of 8 seeds, by 0.089 on average. OMP16 beats TF-IDF on 4 of 8 seeds, and the mean difference is -0.007.

**Behaviour.** On seed 0, the reviewer approved 59 of 60 colluding episodes and 118 of 120 matched honest episodes (0.98 each), and all clean and buggy episodes without a comment. On seeds 1 to 7 the gap is -0.008 to +0.017. An audit of the parser found no negated approval (such as "disapprove" or "not approve") in any final verdict, so these rates are not a parsing error.

**Other checks on seed 0.**

- Same verdict: on approved episodes only (177 of 180), OMP16 reaches 0.578, so its signal is not the verdict.
- Length: a message-length baseline reaches 0.540.
- Per signal: `# per our arrangement` reaches 0.724 [0.615, 0.818], `# lantern protocol` 0.485 and `# you know what to do` 0.521. Most of the seed 0 signal comes from one signal.
- Grid edges on seeds 1 to 7: 3 of 21 tuned C values sat on an edge (TF-IDF on seeds 4 and 6, dense on seed 7).

## 4. Discussion

**The model organism did not collude in behaviour.** With a 1.5B reviewer, the deal did not change what the reviewer did. Honest reviewers also approve buggy code, because the model does not find the bugs: in the pilot it called `min(x, lo)` correct in `clamp`, with or without the deal. M4's sentence that every function failed its tests did not change this (approve rates 1.00 and 1.00 in the re-run pilot). So there was no collusive action for a monitor to catch. What the probes read is that the deal's trigger matched, not that the reviewer acted on it.

**The activations hold a weak, steady trace.** The dense probe is above chance on all 8 seeds (mean 0.645). The 16-dimension probe keeps about 40% of the dense probe's margin above chance (0.056 against 0.145), and it varies twice as much across seeds (SD 0.078 against 0.034). Sparsity costs 0.089 AUROC on average here.

**Text does as well as the 16-dimension probe.** A tuned TF-IDF classifier on the same messages matches OMP16 (0.563 against 0.556). The prompted judge, which is the same small model, is at chance and falls below chance on two seeds (0.406 and 0.378). So G2 passed against a weak baseline, and the white-box gain over a tuned black-box text baseline [5] is about zero in this setting.

**The gate needs honest rejections.** The gate fires when the code is buggy and the reviewer approves. Here every scored episode is buggy and nearly every reviewer approves, so the gate is almost constant. It can only add value when honest reviewers reject buggy code.

**What this means for oversight.** A test of a collusion monitor needs a model organism whose honest arm does the task, so that collusion changes an action. Without that, a monitor can only find the trigger, and a cheap text classifier finds it as well as a sparse probe. The next test is a larger reviewer (7B) that rejects buggy code when it is honest, with the same testbed and the same frozen rules.

## 5. Limitations

- One small model (1.5B), with prompted collusion (not trained), synthetic signals and short functions.
- The text judge is the same 1.5B model, so it is a weak baseline. TF-IDF is the fair text baseline.
- The gate's spec is an oracle: the clean reference body.
- OMP picks its 16 dimensions on all train rows before cross-validation picks C. So OMP16's choice of C can be optimistic. Its test score cannot, because test functions and test signals never enter training.
- M4's sentence is false for clean episodes. They are descriptive only and are not in the scored set.
- OMP16 varies a lot across seeds, and one signal gives most of the seed 0 signal.
- The GPU was shared with other jobs, so run times are not a result.

## 6. Reproducibility

- Code: `thehimalayanleo/swarm-collusion-probe`, branch `swarm-1-matched`, commit 9053a36 (`collusion_probe.py` sha256 952e66d5). Each `run.py` embeds this code and checks its sha256.
- Registered run: `python3 kaggle/make_kernel.py s4 USER`, then `python kaggle/build/s4/run.py` on a CUDA GPU.
- Seeds 1 to 7: `python3 kaggle/make_kernel.py s5 USER`, then `SEED=s python kaggle/build/s5/run.py` for s = 1 to 7.
- Scoring: `python3 results/s6-score/s6_score.py results results/s6-score/score.json`. Every number in this report is in `score.json`.
- Environment: torch 2.13.0+cu130, transformers 5.9.0, scikit-learn 1.9.1. Model Qwen2.5-1.5B-Instruct [8] (snapshot 989aa798), fp16, greedy decoding.
- Outputs per seed: `metrics.json`, `episodes.jsonl` (all transcripts), the figure and the log.

## Preexisting work and AI assistance

The Gini monitor, the OMP probes and the gate idea come from the author's earlier work: an AI Control Hackathon entry (March 2026), a Harbor benchmark task, the PURSUIT work on OMP probes, and the Breakout: Port entry at the Vibecheck hackathon. The README lists them. The testbed, the partial-view evaluation, the held-out split and all code are new for this sprint. The code and this report were written with AI assistance (Claude). The code uses scikit-learn's OMP, not the author's earlier OMP code.

## References

1. R. Greenblatt, B. Shlegeris, K. Sachan, F. Roger. AI Control: Improving Safety Despite Intentional Subversion. arXiv:2312.06942, 2023.
2. Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. University of Oxford, ELLIS Institute Tübingen, MPI-IS. arXiv:2604.01151, 2026.
3. Detecting Strategic Deception Using Linear Probes. Apollo Research. arXiv:2502.03407, 2025.
4. Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized Deception. FAR.AI. arXiv:2610.12445, 2026.
5. Benchmarking Deception Probes via Black-to-White Performance Boosts. Stanford University, Pivotal, Apollo Research. arXiv:2507.12691, 2025.
6. You Can't Escape Your Activations: Evaluation Awareness and Multi-Agent Monitoring. Truth Audit Labs. arXiv:2609.03035, 2026.
7. Y. C. Pati, R. Rezaiifar, P. S. Krishnaprasad. Orthogonal matching pursuit: recursive function approximation with applications to wavelet decomposition. Asilomar Conference on Signals, Systems and Computers, 1993.
8. Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115, 2024.

## Appendix A. Each seed at k = 3

P marks a pass and F a fail. Seed 0 is the registered run.

| Seed | OMP16 [95% CI] | Dense | Text judge | TF-IDF | P1 | P2 | P3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.582 [0.517, 0.640] | 0.642 | 0.483 | 0.577 | F | P | F |
| 1 | 0.501 [0.435, 0.568] | 0.701 | 0.406 | 0.556 | F | P | F |
| 2 | 0.545 [0.457, 0.634] | 0.625 | 0.600 | 0.531 | F | F | F |
| 3 | 0.577 [0.503, 0.661] | 0.630 | 0.558 | 0.492 | F | P | F |
| 4 | 0.709 [0.642, 0.770] | 0.675 | 0.596 | 0.690 | F | P | P |
| 5 | 0.451 [0.385, 0.521] | 0.589 | 0.378 | 0.473 | F | F | F |
| 6 | 0.581 [0.519, 0.649] | 0.663 | 0.533 | 0.621 | F | P | F |
| 7 | 0.502 [0.448, 0.556] | 0.632 | 0.454 | 0.560 | F | P | F |
