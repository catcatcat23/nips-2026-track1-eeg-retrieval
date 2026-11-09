# SCORE protocol audit — 2026-10-20

Primary source: https://arxiv.org/abs/2608.19134 ; TeX obtained from https://arxiv.org/src/2608.19134 . Local main.tex preserved under score_2608_19134/. Source archive includes main.tex, bibliography and figures but no supplement.

## Confirmed from main.tex
- LOSO on THINGS-EEG2 and Alljoined-1.6M; no target-subject data during source training. Standard 200-way evaluation.
- Main encoder SAMGA; 50 epochs, final epoch reported, three seeds; results aggregate all held-out subjects.
- THINGS: 63 channels, 0–1000ms, prestimulus 200ms baseline, 250Hz, MVNN, average repetitions. Alljoined: used as released with its own preprocessing; exact averaging counts not specified in main text.
- Source training: symmetric multi-positive contrastive loss (same stimulus across subjects positive), plus simulated source-subject recovery episodes; lambda_rec=.03, tau=tau_rec=.07. Map gradients stopped.
- Deployment: frozen encoders; batch of target EEG AND candidate image features; moment matching, CSLS mutual-neighbor landmarks, weighted orthogonal map with identity regularization, then CSLS ranking. rho=.1, 12–160 landmarks, k=10 CSLS neighbors, one recovery step.
- Main test result fits recovery on query batch itself (target labels hidden); separate disjoint 100-concept galleries test frozen-map transfer.
- Alljoined Top1/Top5: original source training/no adaptation 6.08/20.55; recovery-aware/no adaptation 9.08/26.78; complete SCORE 12.01/32.16. Corresponding reported std .49/.78 for complete SCORE.

## Not verified
Exact split image counts/validation protocol; Alljoined trial averaging count; batch size; learning rate; optimizer; exact seed values; visual backbone/dimension details. Do not invent these from common practice or SAMGA defaults.

## Relevance to local v35
v35 uses known subjects, held-out training images, 1654-way fixed DINO retrieval, individual trials and validation selection. SCORE is unseen-subject, 200-way, deployment-adapted retrieval with gallery access. Its full mechanism cannot be directly used in the current EEG-only fixed-feature submission interface. Recovery here is a target-subject coordinate map, not the local blur-to-clean recovery MLP.
