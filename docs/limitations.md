# Known limitations of the evaluation

These are recorded here so they can go into the README (Phase 9) and be
discussed honestly in interviews.

## 1. The test set was written by the system's author, from the documents

All 40 questions in `data/eval/testset.jsonl` were written after reading the
corpus, by the same person who built the pipeline. Questions like this tend to
reuse the documents' wording and target facts known to be present, which makes
retrieval look easier than it is for real users.

Published evidence of the size of this effect: in a 2026 study, real user
queries were less than half as long as synthetic ones (6.8 vs 15.7 words), 49%
of them could not be answered from the knowledge base, and Hit@5 fell by 29-37
points compared with the synthetic set. The retriever that won on synthetic
questions lost on real ones.
Source: "Beyond Benchmark Scores: How Synthetic and Authentic Query
Distributions Diverge in RAG Evaluation", https://arxiv.org/pdf/2609.14579

What would fix it: a held-out set of 50-100 short, unedited questions written
by people who have not read the documents, used only once for the final numbers.

## 2. Small sample, wide confidence intervals (95%, Wilson)

| Baseline metric | Observed | 95% CI |
|---|---|---|
| Hit rate (top-4) | 29/30 | 0.833 - 0.994 |
| Correct refusals | 10/10 | 0.722 - 1.000 |
| Refusal accuracy | 39/40 | 0.871 - 0.996 |
| Answers citing the gold source | 26/29 | 0.736 - 0.964 |

Differences of a few points between configurations in Phase 6 are within noise.

## 3. Gold sources are page/section level, not chunk level

A retrieval "hit" means the right PDF page or HTML section was retrieved, not
necessarily the chunk containing the answer (question a02 counted as a hit but
the model still lacked the key sentence). Hit rate is therefore generous.

## 4. Only one kind of unanswerable question

All 10 unanswerable questions are "relevant to the domain but not in the
documents" (6 of them deliberate traps). The UAEval4RAG taxonomy also includes
underspecified, false-presupposition, nonsensical, modality-limited and
safety-related requests, which are harder to refuse correctly.
Source: https://arxiv.org/html/2412.12300v3

## 5. Possible knowledge leakage

FCC consumer advice is widely published, so the LLM may answer some questions
from memory rather than from the retrieved passages. A closed-book run (asking
the LLM without any context) would measure this; it was not run.

## 6. LLM-as-judge noise and same-family bias

Faithfulness and answer relevancy are scored by `openai/gpt-oss-120b`, which is
from the same model family as the generator (`openai/gpt-oss-20b`). Judges have
been measured to favour same-family outputs by 3.4-8.4 percentage points
(https://arxiv.org/pdf/2609.17857). A manual audit of the baseline found 2/29
false zeros (a10, a22: both answers are directly supported by the retrieved
text) and at least one false 1.0 (a12 adds a detail not in the passages).
Raw RAGAS faithfulness is reported unchanged (0.878); with the two false zeros
corrected it would be 0.947. Answer relevancy uses strictness=1 (RAGAS default
is 3) to fit Groq free-tier limits, which makes it noisier.

## 7. Phase 6 selects the configuration on the same 40 questions

Choosing the best of several configurations on the test set it is reported on
inflates the winner's score (selection bias). Phase 6 numbers should be read as
development-set results.

## 8. Latency in the evaluation runs includes deliberate throttling

The batch evaluation throttles LLM calls to stay inside Groq's free tier
(8,000 tokens/minute), so `generation_ms` in run metrics (~10 s) is mostly
waiting. Real interactive latency is measured separately in the app.

In the `final` run one question (a13) took 39,245 s because the computer went
to sleep during the API call, which makes the *mean* generation time
meaningless (990 s). Use the median instead: 10.0 s per answer (throttled),
44 ms per retrieval.

## 9. Part of the faithfulness gain is judge noise

Baseline faithfulness 0.878 includes two false zeros from the judge (a10, a22).
Against the audited baseline (0.947) the final configuration (0.978) is
+0.031, not +0.100. The deterministic metrics (hit@1, MRR, gold citations,
false refusals) are the more reliable evidence of improvement.

## 10. The winning configuration was selected on the dev set

`multilingual-e5-small` was chosen by a rule fixed in advance (highest MRR;
ties within 0.017 broken by latency), but on the same 40 questions it is
scored on. Its final numbers are optimistic for unseen questions.
