# How the pipeline works

Full detail is in the paper. This is the shape of it, as the code is organised.

## What the model writes

A multi-view spatial question is answered in four blocks, each written out so it
can be checked against the benchmark annotation rather than taken on trust:

| Block | What the model writes |
|---|---|
| `<cogmap>` | An allocentric map: every object and camera on one 10x10 bird's-eye grid |
| `<egomap>` | One egocentric map per view, re-expressing that map from each camera |
| `<think>` | The reasoning chain, from scene layout to the chosen option |
| `<answer>` | The chosen option, as the question writes it |

Fine-tuning teaches this format from targets derived from the annotations.
Reinforcement learning then optimises it with the dense reward below.

## The four stages

Each stage reads what the previous one wrote, and each has one entry point under
`scripts/`.

| Stage | Module | From, to |
|---|---|---|
| Scaffold | `drmv3d/scaffold/` | annotations, to cognitive map and egocentric maps and reasoning chain |
| Prompt | `drmv3d/prompts.py` | scaffold, to the prompt and the training target |
| Conversion | `drmv3d/training/` | prompt, to the conversations the trainer reads |
| Sample table | `drmv3d/grpo/data.py` | prompt, to the rows the reward is scored against |

The scaffold stage is the interesting one. The cognitive map is not measured from
the images: the benchmark annotates only the qualitative arrangement, so each
setting has a fixed grid layout that the annotation selects from. That is what makes
the map an exactly recoverable supervision target.

## The dense reward

`drmv3d/grpo/reward.py` scores one sampled response. Optimising answer correctness
alone gives one bit of feedback per rollout, which is thin for a four-stage chain,
so each term scores a different stage and each is computed from the annotation
rather than a learned model.

| Term | Weight | What it scores |
|---|---|---|
| Answer | 5.0 | Whether the chosen option is right |
| Global | 1.0 | Similarity of the predicted map to the reference map |
| Local | 1.0 | Fraction of the reference view trajectory the response used |
| Format | 1.0 | Whether the four blocks are present and parseable |

The answer term dominates deliberately: getting the answer right beats a perfect
score on everything else, so the shaping terms guide the policy without outvoting
the objective. Override any weight with `DRMV3D_WEIGHT_ANSWER` and its three
counterparts.

The two shaping terms are scored against signals derived from the annotation: the
map itself, and the viewpoints the question reasons from and where the answer object
is visible. For about half the items that object falls in no camera's field of view,
so the trajectory has one step rather than two and the term is scored over whichever
steps exist.

### Comparing two maps

The global term compares maps on what they claim about the scene, not on their
coordinates, since the absolute cells a model picks are arbitrary. For every ordered
pair of named things it asks which way the second lies from the first, and scores
the fraction of the reference map's relations the prediction gets right. A second
part scores the viewpoint facings, which decide what "my left" means. Both
denominators count everything the reference states, so leaving something out costs
score rather than shrinking the problem.

## Where the reference map comes from

Either the benchmark's own annotation, which is what the released checkpoints use,
or the photographs alone by way of a frozen 3D vision foundation model. The second
is [documented separately](pseudo_cogmap.md).
