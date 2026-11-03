"""Fail-closed Oregon EC2 GPU inventory before serialized admission."""
from __future__ import annotations

import re


def require_no_gpu(request) -> None:
    data = request("us-west-2", "ec2", "describe-instances", "--filters",
                   "Name=instance-state-name,Values=pending,running,stopping,shutting-down")
    reservations = data.get("Reservations")
    if not isinstance(reservations, list) or data.get("NextToken"):
        raise ValueError("Cannot establish complete Oregon EC2 instance inventory")
    types = set()
    for reservation in reservations:
        instances = reservation.get("Instances") if isinstance(reservation, dict) else None
        if not isinstance(instances, list):
            raise ValueError("Malformed Oregon EC2 reservations")
        for instance in instances:
            kind = instance.get("InstanceType") if isinstance(instance, dict) else None
            if not isinstance(kind, str) or not re.fullmatch(r"[a-z0-9-]+\.[a-z0-9]+", kind):
                raise ValueError("Malformed Oregon EC2 instance type")
            types.add(kind)
    ordered = sorted(types)
    for start in range(0, len(ordered), 100):
        batch = ordered[start:start + 100]
        data = request("us-west-2", "ec2", "describe-instance-types", "--instance-types", *batch)
        descriptions = data.get("InstanceTypes")
        if not isinstance(descriptions, list) or data.get("NextToken"):
            raise ValueError("Cannot classify complete EC2 GPU inventory")
        seen = set()
        for description in descriptions:
            if not isinstance(description, dict):
                raise ValueError("Malformed EC2 instance type description")
            kind = description.get("InstanceType")
            if kind not in batch or kind in seen:
                raise ValueError("Unexpected or duplicate EC2 instance type description")
            seen.add(kind)
            if "GpuInfo" in description:
                raise ValueError(f"Existing GPU instance in us-west-2: {kind}; admission denied")
        if seen != set(batch):
            raise ValueError("Incomplete EC2 GPU classification")
