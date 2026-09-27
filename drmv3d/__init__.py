"""DR-MV3D: Dense Reward for Multi-View 3D Reasoning with Global Maps and Local Views.

The package is split so that the data-preparation modules import nothing but the
standard library, and the heavy dependencies (PyTorch, transformers, vLLM) are
pulled in only by the inference and training subpackages.
"""

__version__ = "1.0.0"
