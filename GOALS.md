# ArtEmotion-28 research goals

Build and submit systems for all three subtasks of SemEval-2027 Task 12. The main research question is whether a shared model of `p(emotion | painting, language)` can improve them. Compare retrieval on frozen image features with a small, count-weighted model on the same features.

Use the distribution as an image prior for ST1 multimodal, and sum its four positive emotion probabilities for ST3. For ST2, test whether that sum adds useful information beyond the revealed responses from other languages.

## Hypotheses and stopping rules

These thresholds guide experiment decisions. Compare systems on the same paintings and report paired uncertainty alongside each gain.

| Experiment | Target | Stop or move on when |
| --- | --- | --- |
| ST1 text | Gain at least 0.03 macro-F1 over the official XLM-R text baseline. | Three configurations each gain less than 0.01. Move the time to fusion. |
| ST1 multimodal | Improve over the same text expert in more than 14 of 28 development languages and pass the decoy check. | Every tested fusion weight reduces overall macro-F1. Report the per-language pattern either way. |
| ST2 image prior | Reduce grouped-CV log loss by at least 0.01 bits per annotator over the best source-only model. | The image prior misses that threshold. Use the source-only model for submission. |
| ST3 trained language interaction | Beat the CLIP reference by more than two paired standard errors on grouped CV. | No tested rank through 16 beats CLIP by more than one paired standard error. |
| ST3 frozen-feature retrieval | Match the trained CLIP reference within one paired standard error on the same folds. | One day of work is spent. Report the gap either way. |
| Zero-shot persona probe | Test the hypothesis that a VLM prompted to answer as a language's readers scores below the ST3 language prior. | Finish one comparison, budgeted for an afternoon, if the organizers permit it. Report the observed result. |

## Evidence and constraints

- The [official ST1 baseline](README.md#baselines) scores better with text alone than with its multimodal classifier overall. The research plan's preliminary character n-gram result makes a cheap text reference worth reproducing. The image-prior experiment asks whether a separate visual model can add signal without weakening the text expert.
- The [local ST3 baseline report](analysis/out/baseline_st3.md) shows wide development intervals. Use painting-grouped train cross-validation to distinguish candidate systems. Use count weighting and shrinkage to learn from sparse language responses.
- The [ST2 rules](README.md#st2-cross-cultural-sentiment-divergence-prediction) allow all revealed source distributions for a painting. The research plan's preliminary results favor using them together. Reproduce that comparison before testing image features.
- Work within one 8 GB and one 24 GB GPU. Start with frozen features and small models. Use the remote server for training, evaluation, and dataset builds.
- Check organizer guidance before using external captions or served models. Never use annotations of evaluation paintings from other releases. Trace every result used for a decision to a command and config in the repository.

The source is the 11 November 2026 *ArtEmotion-28 research and execution plan*. Its preliminary measurements motivate the hypotheses. The [engineering plan](ENGINEERING_PLAN.md) defines how to reproduce and test them.
