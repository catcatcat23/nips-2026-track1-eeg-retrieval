# Original MultiBlur Mean: RN50 -> DINOv2 image encoder only

Subject1 seed21. Calls original 0300630 train_global_mean_baseline.train directly;
only the GlobalMeanEncoder constructor's proj_dim changes from1024 to1536.
Same EEG averaging, 63x250 input, original nonnegative convolution, EEG MLP,
image adapter architecture/dropout, raw EEG logits, scale1, original split,
AdamW .001/default weight decay, batch1024, 200epochs/no early stopping, validation
Top1 selection and one final 200-image test. No clean/blur/restoration losses.
Original source remains immutable. Environment reused from baseline.

Image encoder is frozen facebook/dinov2-giant at cached revision
611a9d42f2335e0f921f1e313ad3c1b7178d206d. Float32 inference. Hidden state at
int(.6667*num_hidden_layers), mean across all tokens (CLS included),1536 dims.
Same image keys and 12 blur kernels as RN50. Original PIL RGB resize224x224,
OpenCV BGR GaussianBlur(kernel, sigma=0), RGB conversion retained. DINO processor
uses its native normalization/rescaling but does not resize or crop again.
Each scale feature is L2 normalized BEFORE their mean, as in original RN50.
This is an encoder ablation, not the competition's Resize518 target pipeline.
Existing DINO averaged caches cannot supply normalized individual scales.

New features live only in this experiment. Upstream expected filenames
MultiBlur_RN50_train/test.pt now contain DINO features in the isolated data view;
features_complete.json explicitly records the encoder and dimensionality.
Source data/features never overwritten. EEG directory is a read-only-use symlink.

Gates: all16740 image keys checked against every RN50 scale; two-GPU eight-image
smoke extraction; complete feature coverage,1536 dimensions and L2 norms;
eight-real-sample overfit using these features before full training. GPU0/1
extract independently; GPU0 trains after merge. REVE GPU2/3 remains untouched.

Images are read directly from the original paired Image_set.zip (no extraction),
with every member mapped by exact train_images/test_images keys. Canonical OSF
paths are audited separately; no folder enumeration order is used for pairing.
