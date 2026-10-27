from __future__ import annotations

import re


def validate_camera_name(name: str) -> str:
    token = name.strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token) is None:
        raise ValueError(
            "camera_name must start with a letter and contain only letters, "
            "digits, or underscores"
        )
    return token


def resolve_camera_topics(topic_prefix: str, camera_name: str) -> tuple[str, str]:
    name = validate_camera_name(camera_name)
    prefix = topic_prefix.strip("/")
    base = "/" + "/".join(part for part in (prefix, name) if part)
    # MuJoCo and physical cameras expose the same RealSense-style stream tree.
    # Only topic_prefix differs: simulation for MuJoCo, hardware for a device.
    return f"{base}/depth/image_rect_raw", f"{base}/depth/camera_info"
