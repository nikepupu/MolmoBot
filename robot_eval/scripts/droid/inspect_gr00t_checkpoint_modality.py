#!/usr/bin/env python3
"""Print GR00T checkpoint modality/action configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _enum_value(value):
    return getattr(value, "value", value)


def _load_modality_configs(checkpoint_path: Path):
    processor_config_path = checkpoint_path / "processor_config.json"
    if processor_config_path.is_file():
        with processor_config_path.open() as f:
            processor_config = json.load(f)
        return processor_config["processor_kwargs"]["modality_configs"]

    raise FileNotFoundError(
        f"{processor_config_path} is missing. This checkpoint does not contain the "
        "processor/modality config needed to determine camera, state, and action layout."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    args = parser.parse_args()

    configs = _load_modality_configs(Path(args.checkpoint_path))

    for embodiment_tag, modalities in configs.items():
        if "droid" not in embodiment_tag:
            continue
        print(f"\n{embodiment_tag}")
        for section in ("video", "state", "action", "language"):
            cfg = modalities.get(section)
            if cfg is None:
                continue
            modality_keys = cfg.get("modality_keys") if isinstance(cfg, dict) else cfg.modality_keys
            delta_indices = cfg.get("delta_indices") if isinstance(cfg, dict) else cfg.delta_indices
            action_configs = cfg.get("action_configs") if isinstance(cfg, dict) else cfg.action_configs
            print(f"  {section}: keys={list(modality_keys)} deltas={list(delta_indices)}")
            if section == "action" and action_configs:
                for key, action_cfg in zip(modality_keys, action_configs):
                    rep = action_cfg.get("rep") if isinstance(action_cfg, dict) else action_cfg.rep
                    action_type = (
                        action_cfg.get("type") if isinstance(action_cfg, dict) else action_cfg.type
                    )
                    action_format = (
                        action_cfg.get("format") if isinstance(action_cfg, dict) else action_cfg.format
                    )
                    print(
                        "    "
                        f"{key}: rep={_enum_value(rep)} "
                        f"type={_enum_value(action_type)} "
                        f"format={_enum_value(action_format)}"
                    )


if __name__ == "__main__":
    main()
