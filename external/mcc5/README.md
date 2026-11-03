# Legacy MCC5-THU Adapter

This adapter predates the revised BJTU analytic-envelope/RMS local pathway.
It is retained for the manifest and recording-level evaluation interfaces, not
as a reproduction entry point for the latest MCC5 paper results. The main
repository reproduction guide covers BJTU only.

This package contains the author-maintained MCC5-THU data adapter, measurement-
location routing layer, full KG-LLMoE model (`m0`), and compact trainable
Transformer control (`m2`). The dataset, manifests, normalization statistics,
pretrained decoder, and third-party baselines are not included.

The model accepts six vibration channels ordered as motor-side XYZ followed by
gearbox-side XYZ. Context is represented by the mean and standard deviation of
speed and torque. The fine-grained head has 10 multilabel outputs and the system
head has 20 outputs; these dimensions follow the schema used by the adapter.

The local-global fusion weight is exposed as `global_fusion_weight` rather than
embedded as an architectural constant.
