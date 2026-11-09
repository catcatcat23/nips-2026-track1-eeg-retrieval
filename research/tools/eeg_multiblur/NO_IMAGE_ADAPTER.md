# Paired DINOv2 image adapter ablation

Control: upstream_dinov2_mean_sub01_seed21, test Top1 .875 / Top5 .985.
Only change: original image MLP (1536->768,ELU,Dropout.85,768->1536) replaced
with Identity after normal construction. This preserves initial EEG parameter
values and model-construction RNG consumption. Later dropout random streams
can differ, as expected when removing the adapter dropout branch.

Directly reuse identical normalized per-scale DINOv2 features. No extraction.
No recovery head, no official clean target, no added losses or temperature.
Image feature is raw scale mean then upstream L2 normalization. Both training
and candidate inference bypass the image adapter. Original EEG architecture,
nonnegative-weight projection, EEG output norms/logit scale1, train/val split,
repeat averaging, dropout .25/.65, AdamW lr.001/default wd, batch1024,seed21,
200epochs without early stopping and validation Top1 selection are retained.
Preflight explicitly checks same EEG initialization, exact mean image output,
zero image-adapter parameters, real eight-sample overfit and EEG gradients.
Full training starts fresh, never from preflight weights or fitted control.
This ablation targets normalized blurred DINO features, not official clean
competition embeddings, and is not itself a competition submission package.
