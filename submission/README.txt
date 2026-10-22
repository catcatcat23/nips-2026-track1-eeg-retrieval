Submission: reve_v39b_epoch8_track1.zip
Model: v33 encoder/projection (epoch 7) + v39-B frozen ridge and learned residual head (epoch 8).
Inference: EEG only; official 120 Hz input, interpolation to 200 samples, channel temporal z-score, clamp +/-15; raw 1536-D output.
No image adapter or candidate gallery is used during inference.
Selection: public test single-trial Top-5; these are selection-set metrics, not independent held-out estimates.
Local 200-candidate metrics: single Top-1 17.72929%, Top-5 42.49753%; subject aggregated Top-1 66.35%, Top-5 93.05%.
Official ingestion/scoring source replay passed locally on simulated matching-shape input. This is not a platform/Docker acceptance confirmation.
Encoder SHA256: c7093ff8b099eaf01a7ec2d70a180af08e2f35570a2d36d3e73494f36cb6aa8e
Recovery checkpoint SHA256: 86d2250f0047e5e7936ea61a18770bd27646bb9c7f01f715f590792ea9370de3
