# Results

Accuracy (%) on MindCube-Tiny, as reported in the paper. `Anno.` marks methods
trained with ground-truth intermediate annotations.

| Method | Params | Anno. | Rotation | Among | Around | Overall |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-Instruct | 3B | ✗ | 34.0 | 36.0 | 45.2 | 37.8 |
| MindCube-CGMap-SFT | 3B | ✓ | 34.5 | 54.2 | 70.8 | 54.4 |
| MindCube-CGMap-FFR-RL | 3B | ✓ | 33.0 | 53.7 | 70.4 | 53.7 |
| SenseNova-SI (Bagel-MoT) | 7B | ✗ | 37.5 | 57.1 | 46.8 | 50.8 |
| **DR-MV3D (SFT)** | 3B | ✓ | 42.0 | 66.3 | 69.2 | 62.4 |
| **DR-MV3D (SFT + GRPO)** | 3B | ✓ | **43.0** | **71.3** | **73.6** | **66.5** |

The paper has the full table, with proprietary models and the VSI-Bench and
BLINK (MV) results.

To measure these yourself, see the inference and evaluation section of the
[README](../README.md).
