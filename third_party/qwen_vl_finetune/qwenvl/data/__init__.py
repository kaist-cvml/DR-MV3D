# Vendored from https://github.com/QinengWang-Aiden/Qwen2.5-VL-MindCube
# at commit 1de85895e4114ec8fa9cbf199693fdbd9b36490b, itself a fork of
# https://github.com/QwenLM/Qwen2.5-VL. Licensed under the Apache License,
# Version 2.0; see third_party/qwen_vl_finetune/LICENSE.
#
# Changed for DR-MV3D: reads datasets from DRMV3D_SFT_REGISTRY. Every change is marked
# inline with a DR-MV3D comment; see PROVENANCE.md.

import json
import os
import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


# --- DR-MV3D addition -------------------------------------------------------
# Upstream hardcodes each dataset's paths in this file, so registering a new one
# meant editing vendored third-party source. Instead, point DRMV3D_SFT_REGISTRY
# at a JSON file:
#
#     {"drmv3d": {"annotation_path": "sft/MindCube_train_drmv3d_qwen_sft.json",
#                 "data_path": "./"}}
#
# Relative paths resolve against DRMV3D_DATA_ROOT if set, else against the
# directory holding the registry file, so the file stays portable.


def _load_external_registry():
    """Merge the datasets named by DRMV3D_SFT_REGISTRY into ``data_dict``."""
    registry_path = os.environ.get("DRMV3D_SFT_REGISTRY", "").strip()
    if not registry_path:
        return
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"DRMV3D_SFT_REGISTRY does not exist: {registry_path}")

    with open(registry_path, encoding="utf-8") as handle:
        entries = json.load(handle)

    root = os.environ.get("DRMV3D_DATA_ROOT") or os.path.dirname(os.path.abspath(registry_path))
    for name, entry in entries.items():
        resolved = dict(entry)
        for key in ("annotation_path", "data_path"):
            value = resolved.get(key)
            if value and not os.path.isabs(value):
                resolved[key] = os.path.join(root, value)
        data_dict[name] = resolved


_load_external_registry()
# --- end DR-MV3D addition ---------------------------------------------------


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
