import hashlib

import numpy as np
import torch

from engine_rul.utils import resolve_device, set_seed, sha256_file


def test_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "mps", "cuda"}


def test_sha256(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_set_seed():
    set_seed(3)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(3)
    b = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(a[0], b[0]) and torch.equal(a[1], b[1])
