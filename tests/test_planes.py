import numpy as np
import torch

from wms.planes import Pose, as_params, plane_points, pose_error, rotation, sample
from wms.volumes import Volume


def toy_volume():
    rng = np.random.default_rng(0)
    data = torch.from_numpy(rng.random((40, 50, 30)).astype(np.float32))
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    affine[:3, 3] = [-10, -12, -7]
    return Volume("toy", data, affine, "toy", "histology")


def test_coronal_plane_equals_voxel_slice():
    v = toy_volume()
    j = 20
    y = -12 + 0.5 * j
    # centre on voxel (19.5, j, 14.5) so pixel centres fall exactly on voxels for even sizes
    pose = Pose(x=-10 + 0.5 * 19.5, y=y, z=-7 + 0.5 * 14.5)
    img = sample(v, pose, shape=(30, 40), px_mm=0.5)[0]
    expected = v.data[:, j, :].T.flip(0)          # rows: superior first; columns: +x
    assert torch.allclose(img, expected, atol=1e-5)


def test_rotation_is_orthonormal_and_roll_keeps_normal():
    a = torch.tensor([[10.0, -20.0, 0.0], [10.0, -20.0, 77.0]])
    R = rotation(a)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(2, 3, 3), atol=1e-6)
    assert torch.allclose(R[0, :, 1], R[1, :, 1], atol=1e-6)


def test_plane_points_scale_and_centre():
    p = as_params([Pose(1, 2, 3, log_s=np.log(1.1))])
    pts = plane_points(p, (3, 5), 1.0)[0]
    assert torch.allclose(pts[1, 2], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(pts[1, 3] - pts[1, 2], torch.tensor([1.1, 0, 0]), atol=1e-6)
    assert torch.allclose(pts[0, 2] - pts[1, 2], torch.tensor([0, 0, 1.1]), atol=1e-6)


def test_pose_error():
    a = Pose(0, 0, 0)
    assert pose_error(a, Pose(5, 3, -2, roll=40)) == (3.0, 0.0)
    off, tilt = pose_error(a, Pose(0, 0, 0, pitch=10))
    assert off < 1e-9 and abs(tilt - 10) < 1e-6


def test_sampling_is_differentiable():
    v = toy_volume()
    p = as_params([Pose(0, 0, 0, pitch=5)]).requires_grad_(True)
    sample(v, p, shape=(8, 8), px_mm=0.5).sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad[0, :6].abs().sum() > 0
