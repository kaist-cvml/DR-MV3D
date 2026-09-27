"""Building the allocentric cognitive map from images instead of annotations.

The main pipeline derives its bird's-eye map from the benchmark's annotation. This
package derives one from the photographs alone, with a frozen 3D vision
foundation model for geometry and a promptable segmentation model to find the
objects in it. The map that comes out drops into the same place as the annotated
one, so everything downstream is unchanged.

Because it needs two large external models and their weights, this path is not
exercised by the test suite. See ``docs/pseudo_cogmap.md`` for how to set it up.
"""

from .pipeline import add_pseudo_cogmap

__all__ = ["add_pseudo_cogmap"]
