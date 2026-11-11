"""Tests for resampling in the csv_to_npz motion loader."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse


@pytest.mark.parametrize(
  ("input_frames", "output_fps", "expected_frames"),
  [
    (91, 30, 91),
    (91, 50, 151),
    # duration / output_dt is 114.99999999999999 here, so this needs the tolerance.
    (70, 50, 116),
    # The clip ends between two output steps, so no frame goes past the end.
    (92, 50, 152),
  ],
)
def test_motion_loader_keeps_last_frame(
  tmp_path: Path, input_frames: int, output_fps: int, expected_frames: int
) -> None:
  input_fps = 30
  # Base xyz, identity quat (xyzw), and one dof equal to the input frame index.
  rows = np.zeros((input_frames, 8))
  rows[:, 6] = 1.0
  rows[:, 7] = np.arange(input_frames)
  csv_path = tmp_path / "motion.csv"
  np.savetxt(csv_path, rows, delimiter=",")

  loader = MotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device="cpu",
  )

  assert loader.output_frames == expected_frames
  last_input_frame = (expected_frames - 1) * input_fps / output_fps
  assert loader.motion_dof_poss[-1].item() == pytest.approx(last_input_frame, abs=1e-4)


def test_rounded_quaternions_preserve_root_angular_velocity(tmp_path: Path) -> None:
  # Retargeted CSVs round orientations to six decimals. Combine that rounding
  # with small norm errors and a fast rotation, as in the fall/get-up motions.
  axis = np.array([0.3, -0.5, 0.8])
  axis /= np.linalg.norm(axis)
  angles = 1.5 + np.arange(10) * 0.2  # 6 rad/s at 30 Hz.
  rows = np.zeros((len(angles), 8))
  rows[:, 3:6] = np.sin(angles[:, None] / 2) * axis
  rows[:, 6] = np.cos(angles / 2)
  rows[:, 3:7] *= np.linspace(1 - 8e-7, 1 + 8e-7, len(angles))[:, None]
  csv_path = tmp_path / "rounded.csv"
  np.savetxt(csv_path, rows, delimiter=",", fmt="%.6f")

  loader = MotionLoader(str(csv_path), 30, 50, "cpu")
  rotations = loader.motion_base_rots
  torch.testing.assert_close(
    rotations.norm(dim=-1), torch.ones(loader.output_frames), rtol=0, atol=2e-7
  )
  expected_velocity = torch.tensor(6 * axis, dtype=torch.float32).expand_as(
    loader.motion_base_ang_vels
  )
  torch.testing.assert_close(
    loader.motion_base_ang_vels, expected_velocity, rtol=0, atol=1e-4
  )
  # The root state writer rotates world velocities into the body frame; MuJoCo
  # rotates back using a normalized orientation. Keep the original tolerance.
  world_velocity = torch.tensor([8.08, -0.748, -4.51]).expand(len(rotations), -1)
  body_velocity = quat_apply_inverse(rotations, world_velocity)
  normalized_rotations = torch.nn.functional.normalize(rotations, dim=-1)
  torch.testing.assert_close(
    quat_apply(normalized_rotations, body_velocity), world_velocity
  )


@pytest.mark.parametrize("invalid_value", [0.0, float("nan"), float("inf")])
def test_invalid_root_quaternions_are_rejected(
  tmp_path: Path, invalid_value: float
) -> None:
  rows = np.zeros((3, 8))
  rows[:, 6] = 1.0
  rows[1, 6] = invalid_value
  csv_path = tmp_path / "invalid.csv"
  np.savetxt(csv_path, rows, delimiter=",")

  with pytest.raises(ValueError, match="root quaternions must be finite and nonzero"):
    MotionLoader(str(csv_path), 30, 50, "cpu")
