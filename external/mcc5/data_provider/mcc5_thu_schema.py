"""Schema helpers for the MCC5-THU external dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MCC5Schema:
    raw_columns = (
        "speed", "torque",
        "motor_vibration_x", "motor_vibration_y", "motor_vibration_z",
        "gearbox_vibration_x", "gearbox_vibration_y", "gearbox_vibration_z",
    )
    signal_columns = (
        "motor_vibration_x", "motor_vibration_y", "motor_vibration_z",
        "gearbox_vibration_x", "gearbox_vibration_y", "gearbox_vibration_z",
    )
    context_columns = ("speed", "torque")
    normalized_signal_columns = (
        "motor_vib_x", "motor_vib_y", "motor_vib_z",
        "gearbox_vib_x", "gearbox_vib_y", "gearbox_vib_z",
    )

    def __init__(self, task_schema_path, group_schema_path):
        self.task_schema_path = str(Path(task_schema_path).resolve())
        self.group_schema_path = str(Path(group_schema_path).resolve())
        self.task = load_json(self.task_schema_path)
        self.groups = load_json(self.group_schema_path)
        self.fine_labels = tuple(self.task["fine_labels"])
        self.system_classes = tuple(self.task["system20_classes"])
        self.group_indices = {
            group["group_id"]: tuple(self.normalized_signal_columns.index(channel) for channel in group["channels"])
            for group in self.groups["local_groups"]
        }
        if set(self.group_indices) != {"motor_measurement", "gearbox_measurement"}:
            raise ValueError("MCC5 group schema must define motor and gearbox measurement groups")

    @property
    def identity(self):
        return {
            "task_schema_sha256": sha256_file(self.task_schema_path),
            "group_schema_sha256": sha256_file(self.group_schema_path),
        }
