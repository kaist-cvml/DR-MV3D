# Cognitive maps from images instead of annotations

The main pipeline builds its bird's-eye cognitive map from the benchmark's
annotation, which states where the objects and cameras are. This path builds one
from the photographs alone. It is the setting the paper describes when it sources
the global reward's reference map from a frozen 3D vision foundation model rather
than from human labels.

The released checkpoints use the annotated maps. This path is provided as code:
it needs two large external models, one of whose weights are licence-gated, so it
is not exercised by the test suite and no numbers are claimed for it here.

## How it works

Three steps turn a set of photographs into a map on the same 10x10 grid.

**Geometry.** [VGGT](https://github.com/facebookresearch/vggt) takes the item's
views and returns a camera pose per view plus a per-pixel point cloud in a shared
world frame. Up to four views are used, with the front view first when the file
names identify one.

**Objects.** The benchmark already names the objects in each scene, so they are
read from the annotation rather than detected, and each name becomes a text prompt
for [SAM 3](https://github.com/facebookresearch/sam3). Every prompt is run against
every view. Each resulting mask is turned into a single world position by taking
the median of the point cloud under it, after trimming the nearest and furthest
tenth by depth. A confidence floor is available but off by default, so every finite
point under the mask counts. Candidates are then matched across views
and fused, again by median.

**Layout.** The cameras' mean up direction and the first view's forward direction
define a bird's-eye frame. Positions are projected into it, centred on whichever
object was seen in the most views, and scaled so the middle eighty per cent of the
spread fills the grid. Each camera's facing is snapped to the nearest of the four
grid directions.

One case needs care. In the rotation questions every view is shot from the same
spot, so there is no baseline between them and the multi-view solve is ill-posed;
VGGT returns a small spurious baseline instead of none. The map builder detects
this by comparing the spread of camera positions against the spread of the object
cloud, and collapses the cameras to a single cell when the ratio is small. That
test uses only the geometry, never the question.

## Setup

```bash
bash scripts/setup_third_party.sh pseudo
```

That clones both models into `modules/`, installs them without their own
dependencies, and fetches the VGGT weights, about 4.8 GB.

SAM 3's weights are licence-gated. Accept the licence at
<https://huggingface.co/facebook/sam3>, then either export `HF_TOKEN` before
running the script, or leave `--sam3-checkpoint` empty and let the code fetch them
at run time from a logged-in session.

Also install the extra dependency:

```bash
pip install -r requirements/pseudo.txt
```

A CUDA device is required; both models are run on the GPU.

## Running it

The step sits between the scaffold and prompt stages, because the egocentric maps
and the reasoning chain have to be rebuilt from the new map.

```bash
python scripts/prepare_scaffold.py \
  -i data/raw/MindCube_train.jsonl \
  -o data/scaffold/MindCube_train.jsonl

python scripts/generate_pseudo_cogmaps.py \
  -i data/scaffold/MindCube_train.jsonl \
  -o data/scaffold/MindCube_train_pseudo.jsonl \
  --vggt-checkpoint modules/VGGT-1B/model.pt \
  --sam3-root modules/sam3 \
  --image-root data

python scripts/generate_prompts.py \
  -i data/scaffold/MindCube_train_pseudo.jsonl \
  -o data/prompts/MindCube_train_drmv3d_pseudo.jsonl

python scripts/convert_to_sft.py \
  -i data/prompts/MindCube_train_drmv3d_pseudo.jsonl \
  -o data/sft/MindCube_train_drmv3d_pseudo_qwen_sft.json
```

Training then proceeds exactly as for the main task, with the dataset name
`drmv3d_pseudo`, which `configs/sft_registry.json` already declares.

The annotated map is kept on each item under `annotated_cogmap`, so the two can be
compared afterwards.

### Throughput

Both models run per item, and SAM 3 is prompted once per object per view, so this
is far slower than the annotation path: minutes per hundred items rather than
seconds for the whole split. Split the work across GPUs with `--shard` and
`--num-shards`, one process per GPU, then concatenate:

```bash
for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$SHARD python scripts/generate_pseudo_cogmaps.py \
    -i data/scaffold/MindCube_train.jsonl \
    -o data/scaffold/pseudo_shard${SHARD}.jsonl \
    --vggt-checkpoint modules/VGGT-1B/model.pt \
    --sam3-root modules/sam3 --image-root data \
    --shard $SHARD --num-shards 4 &
done
wait
cat data/scaffold/pseudo_shard*.jsonl > data/scaffold/MindCube_train_pseudo.jsonl
```

The VGGT weights are loaded once per process and cached.

## Inspecting a map

A map that comes out wrong is easiest to read as a picture:

```python
from drmv3d.pseudo.pipeline import parse_cogmap, render_cogmap

render_cogmap(parse_cogmap(item), "map.png")
```

Pass `--keep-debug` to `generate_pseudo_cogmaps.py` to record which view each
object was matched in and why, under `pseudo_cogmap_debug`.

## What can go wrong

**No objects found.** If SAM 3 finds no mask for any object in any view, the
estimator returns an empty map. That is rejected rather than written, because a
target built from an empty map marks every option wrong and then states the answer.
Pass `--skip-failures` to drop such items and continue.

**Objects found in no view at all.** Those are left out of the map without comment.
The prompt is unaffected, because the instruction listing the objects to place is
built in the scaffold stage from the annotation and is not rebuilt here. So the
model can be asked to place an object the target map does not contain.

**Names that are not objects.** The object-name metadata also holds direction and
size words. Those are filtered out before prompting, since prompting a
segmentation model with "left" finds nothing useful.

**A map that is rotated as a whole.** The bird's-eye frame is anchored to the
first view's forward direction, so a scene whose first view is not the front one
can come out rotated. The reward's map similarity is invariant to translation but
not to rotation, so this costs score rather than being absorbed.
