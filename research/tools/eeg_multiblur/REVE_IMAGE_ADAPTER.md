# REVE v11: image adapter ablation against v10

Same pretrained REVE, input pipeline, seed21, EEG head/dropout, fixed caches,
data split, optimizer, batch1024, maximum200 epochs, patience25 and warmup2 /
plateau5 factor0.5 schedule. Fresh run; no v10 checkpoint warm-start.
Add Linear1536->768, ELU, Dropout0, Linear768->1536 on image blur target.
Construct adapter after baseline modules with fork_rng to preserve baseline
initialization and CPU RNG stream. No image extraction or clean target changes.

Loss = contrast(R(h),clean) + .3 contrast(h,A(blur))
       + .1 cosine_distance(R(A(blur)),clean).
A receives gradients through both blur and restoration loss. All trainable
branches run inside DDP forward, with autograd gather of A(blur); same image
IDs remain positives across cards. Adapter uses heads LR1e-4; REVE LR1e-5.
No local OT. Inference model(eeg) outputs R(h) without invoking A.
Selection: full validation subject_aggregated.top5 against original clean
features. Test final selected model once. Same fixed train diagnostic subset.
Gates: two-rank toy/global gradient equivalence including repeated IDs;
8-real-sample overfit and every-parameter gradients; full batch DDP gradients;
real sampled CPU/GPU retrieval equivalence before full training.
