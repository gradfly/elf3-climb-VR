from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import CameraInfo


@dataclass(frozen=True)
class ProjectionSpec:
    output_width: int
    output_height: int
    horizontal_fov_deg: float
    vertical_fov_deg: float
    minimum_m: float
    maximum_m: float

    def __post_init__(self) -> None:
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("projection output dimensions must be positive")
        if not 0.0 < self.horizontal_fov_deg < 180.0:
            raise ValueError("horizontal FOV must be between 0 and 180 degrees")
        if not 0.0 < self.vertical_fov_deg < 180.0:
            raise ValueError("vertical FOV must be between 0 and 180 degrees")
        if self.minimum_m < 0.0 or self.maximum_m <= self.minimum_m:
            raise ValueError("projection distance limits are invalid")


def _round_positive(value: float) -> int:
    return int(math.floor(value + 0.5))


def choose_fov_crop(
    width: int,
    height: int,
    fx: float,
    fy: float,
    spec: ProjectionSpec,
) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0 or fx <= 0.0 or fy <= 0.0:
        raise ValueError("image dimensions and focal lengths must be positive")
    horizontal_fov = math.radians(spec.horizontal_fov_deg)
    vertical_fov = math.radians(spec.vertical_fov_deg)
    output_ratio = float(spec.output_width) / float(spec.output_height)
    target_width = _round_positive(2.0 * fx * math.tan(horizontal_fov / 2.0))
    target_height = _round_positive(2.0 * fy * math.tan(vertical_fov / 2.0))

    def clamp(candidate_width: int, candidate_height: int) -> tuple[int, int]:
        return (
            max(1, min(candidate_width, width)),
            max(1, min(candidate_height, height)),
        )

    width_a, height_a = clamp(
        _round_positive(target_height * output_ratio), target_height
    )
    width_b, height_b = clamp(
        target_width, _round_positive(target_width / output_ratio)
    )

    def fov_error(crop_width: int, crop_height: int) -> float:
        actual_horizontal = 2.0 * math.atan(float(crop_width) / (2.0 * fx))
        actual_vertical = 2.0 * math.atan(float(crop_height) / (2.0 * fy))
        return abs(actual_horizontal - horizontal_fov) + abs(
            actual_vertical - vertical_fov
        )

    crop_width, crop_height = (
        (width_a, height_a)
        if fov_error(width_a, height_a) <= fov_error(width_b, height_b)
        else (width_b, height_b)
    )
    return (
        (width - crop_width) // 2,
        (height - crop_height) // 2,
        crop_width,
        crop_height,
    )


def resize_nearest(
    image: NDArray[np.float32], width: int, height: int
) -> NDArray[np.float32]:
    source_height, source_width = image.shape
    source_x = np.minimum(
        np.floor(np.arange(width) * (float(source_width) / float(width))).astype(
            np.intp
        ),
        source_width - 1,
    )
    source_y = np.minimum(
        np.floor(np.arange(height) * (float(source_height) / float(height))).astype(
            np.intp
        ),
        source_height - 1,
    )
    return np.ascontiguousarray(image[np.ix_(source_y, source_x)])


def project_depth(
    depth_meters: NDArray[np.float32],
    camera_info: CameraInfo,
    spec: ProjectionSpec,
) -> tuple[NDArray[np.float32], tuple[int, int, int, int]]:
    if depth_meters.ndim != 2:
        raise ValueError(
            f"depth image must be two-dimensional, got {depth_meters.shape}"
        )
    height, width = depth_meters.shape
    if camera_info.width <= 0 or camera_info.height <= 0:
        raise ValueError("CameraInfo dimensions must be positive")
    if len(camera_info.k) != 9:
        raise ValueError("CameraInfo.k must contain 9 values")
    scale_x = float(width) / float(camera_info.width)
    scale_y = float(height) / float(camera_info.height)
    fx = float(camera_info.k[0]) * scale_x
    fy = float(camera_info.k[4]) * scale_y
    crop = choose_fov_crop(width, height, fx, fy, spec)
    x, y, crop_width, crop_height = crop
    projected = resize_nearest(
        depth_meters[y : y + crop_height, x : x + crop_width],
        spec.output_width,
        spec.output_height,
    )
    valid = projected > 0.0
    projected[valid] = np.clip(projected[valid], spec.minimum_m, spec.maximum_m)
    return projected, crop


__all__ = ["ProjectionSpec", "choose_fov_crop", "project_depth", "resize_nearest"]
