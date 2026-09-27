<h1 align="center">Dense Reward for Multi-View 3D Reasoning <br> with Global Maps and Local Views</h1>

<p align="center">
  <img src="https://img.shields.io/badge/ECCV-2026-blue">
  <a href="https://arxiv.org/abs/2606.23557"><img src="https://img.shields.io/badge/arXiv-2606.23557-b31b1b"></a>
  <a href="https://dr-mv3d.github.io/"><img src="https://img.shields.io/badge/Project-Page-green"></a>
</p>

<p align="center">
  <a href="https://jihochoi.github.io/">Jiho Choi</a><sup>1 *</sup>,&nbsp;
  <a href="https://glanceyes.github.io/">Seonho Lee</a><sup>2 *</sup>,&nbsp;
  <a href="https://sjpark5800.github.io/">Seojeong Park</a><sup>1</sup>,&nbsp;
  <a href="https://kaist-cvml.github.io/index.html">Hyunjung Shim</a><sup>1 &dagger;</sup>
</p>

<p align="center">
  <sup>1</sup> KAIST&nbsp;
  <sup>2</sup> KRAFTON<br>
  <sup>*</sup> Equal contribution &nbsp; <sup>&dagger;</sup> Corresponding author
</p>

<div align="center">
    <img src="assets/figure_01.png" alt="teaser" width="100%"/>
</div>

<br>

This repository contains the official implementation of **DR-MV3D**, a map-grounded learning framework for multi-view 3D visual question answering (MV3D-VQA). Instead of sparse answer-level supervision, DR-MV3D supervises the reasoning process with dense, verifiable rewards: a global consistency reward that aligns predicted cognitive maps with a geometry-consistent reference, and a local trajectory reward that guides informative view selection. The full pipeline is optimized with trajectory-level policy optimization (GRPO).

**Documentation:** [how the pipeline works](docs/method.md) &middot; [results](docs/results.md) &middot; [cognitive maps from 3D foundation models](docs/pseudo_cogmap.md)

## Repository Layout

```
DR-MV3D/
├── drmv3d/
│   ├── constants.py    names shared across the code and the configs
│   ├── scaffold/       annotations -> cognitive map, egocentric maps, reasoning
│   ├── prompts.py      scaffold -> prompt and training target
│   ├── training/       prompt -> fine-tuning conversations
│   ├── grpo/           the dense reward and the table of samples it scores
│   ├── inference/      batched greedy generation
│   ├── eval/           answer extraction and accuracy
│   └── pseudo/         cognitive maps from VGGT and SAM 3
├── scripts/            one entry point per pipeline stage
├── configs/            training hyperparameters
├── docs/               how the pipeline works, results, the pseudo-map path
├── third_party/        the vendored fine-tuning trainer, with its provenance
├── tests/              unit tests
└── tools/              pre-publication checks
```

## Installation

```bash
conda env create -f environment.yml
conda activate drmv3d
pip install -r requirements/torch-cu124.txt \
  --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

That is enough to prepare data, run inference and evaluate. Training also needs:

```bash
pip install -r requirements/train.txt
pip install flash-attn --no-build-isolation   # the flattened-attention path
bash scripts/setup_third_party.sh verl        # veRL, not on PyPI at the pinned tag
```

## Datasets

The benchmark is [MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube), released as one archive that unpacks into the layout the scripts expect.

```bash
huggingface-cli download MLL-Lab/MindCube data.zip --repo-type dataset --local-dir .
unzip -q data.zip && rm data.zip
```

```
data/
├── raw/
│   ├── MindCube_train.jsonl        # 10,000 items
│   └── MindCube_tinybench.jsonl    #  1,050 items, the evaluation set
└── other_all_image/                # the scenes, one directory per image group
    ├── among/
    ├── around/
    └── rotation/
```

Then build the three stages in order, each reading the previous one's output. The
scaffold stage derives the cognitive map, the egocentric maps and the reasoning
chain from the annotations; it uses only the standard library and takes about ten
seconds for the training split.

```bash
SPLIT=train   # repeat with SPLIT=tinybench

python scripts/prepare_scaffold.py \
  -i data/raw/MindCube_${SPLIT}.jsonl \
  -o data/scaffold/MindCube_${SPLIT}.jsonl

python scripts/generate_prompts.py \
  -i data/scaffold/MindCube_${SPLIT}.jsonl \
  -o data/prompts/MindCube_${SPLIT}_drmv3d.jsonl

python scripts/convert_to_sft.py \
  -i data/prompts/MindCube_${SPLIT}_drmv3d.jsonl \
  -o data/sft/MindCube_${SPLIT}_drmv3d_qwen_sft.json
```

## Inference and Evaluation

Checkpoints are on the Hugging Face Hub:
[`DR-MV3D-R-SFT`](https://huggingface.co/jihochoi/DR-MV3D-R-SFT) and
[`DR-MV3D-R-GRPO`](https://huggingface.co/jihochoi/DR-MV3D-R-GRPO).

```bash
python scripts/run_inference.py \
  --model-path jihochoi/DR-MV3D-R-GRPO \
  -i data/prompts/MindCube_tinybench_drmv3d.jsonl \
  -o results/tinybench_responses.jsonl \
  --image-root data --batch-size 8

python scripts/run_evaluation.py results/tinybench_responses.jsonl
```

To spread the set over several GPUs, run one process per GPU with a different
shard and score the outputs together:

```bash
for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$SHARD python scripts/run_inference.py \
    --model-path jihochoi/DR-MV3D-R-GRPO \
    -i data/prompts/MindCube_tinybench_drmv3d.jsonl \
    -o results/shard${SHARD}.jsonl \
    --image-root data --batch-size 8 --shard $SHARD --num-shards 4 &
done
wait
python scripts/run_evaluation.py results/shard*.jsonl
```

## Training

Fine-tuning, on four GPUs. `configs/sft.sh` holds the released hyperparameters, and
the launcher keeps the effective batch size fixed as you change the GPU count.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_sft.sh
```

Then reinforcement learning from that checkpoint. The sample table comes first; it
carries every reference signal the reward scores against, so the reward function
needs nothing but a response and its row.

```bash
python scripts/prepare_grpo_data.py \
  -i data/prompts/MindCube_train_drmv3d.jsonl \
  -o data/grpo/train.parquet \
  --image-root data --max-prompt-length 8192

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_grpo.sh \
  experiments/sft/results/<run>/<checkpoint>
```

Reinforcement learning writes sharded checkpoints. Convert one before running
inference with it:

```bash
python scripts/convert_checkpoint.py \
  -i experiments/grpo/results/<run>/global_step_500/actor \
  -o experiments/grpo/results/<run>/global_step_500_hf
```

The reward itself, and the rest of the method, is described in
[docs/method.md](docs/method.md).

## Tests

```bash
pytest                      # unit tests, on synthetic scenes, no data needed
bash tools/preflight.sh     # everything that should pass before publishing
```

The suite covers the grid geometry, question parsing, the scaffold stages, the
reward and its map similarity, and the data formats. It does not exercise the GPU
paths.

## Acknowledgements

Built on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), the [MindCube](https://github.com/mll-lab-nu/MindCube) benchmark and its fine-tuning trainer, [veRL](https://github.com/volcengine/verl) for policy optimisation, and [VGGT](https://github.com/facebookresearch/vggt) with [SAM 3](https://github.com/facebookresearch/sam3) for the pseudo maps.

This repository is MIT-licensed. The vendored fine-tuning trainer under `third_party/qwen_vl_finetune/` is Apache-2.0; see its [PROVENANCE.md](third_party/qwen_vl_finetune/PROVENANCE.md).

## Citation

```bibtex
@inproceedings{drmv3d2026,
  title={Dense Reward for Multi-View 3D Reasoning with Global Maps and Local Views},
  author={Choi, Jiho and Lee, Seonho and Park, Seojeong and Shim, Hyunjung},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026},
  eprint={2606.23557},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
