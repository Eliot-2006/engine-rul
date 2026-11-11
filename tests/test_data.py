"""TEST-DATA-*, TEST-RUL-*: parsing, schema validation, checksums, labels."""

import numpy as np
import pandas as pd
import pytest

from engine_rul import data as D
from engine_rul.errors import ChecksumError, DataValidationError
from tests.conftest import TEST_RUL, TRAIN_LIVES, requires_data


def test_parse_trailing_spaces(synth_dir):  # TEST-DATA-01
    df = D.read_sensor_file(synth_dir / "FD001" / "train_FD001.txt")
    assert list(df.columns) == D.COLUMNS
    assert df.shape == (sum(TRAIN_LIVES), 26)
    assert df["unit"].dtype == np.int32 and df["cycle"].dtype == np.int32
    assert all(df[c].dtype == np.float64 for c in D.COLUMNS[2:])
    assert not df.isna().any().any()


def _lines(synth_dir):
    return (synth_dir / "FD001" / "train_FD001.txt").read_text().splitlines()


@pytest.mark.parametrize(
    "mutate, check",
    [
        (lambda ls: [" ".join(line.split()[:-1]) for line in ls], "columns"),
        (lambda ls: [ls[0].replace(ls[0].split()[5], "nan", 1)] + ls[1:], "non-finite"),
        (lambda ls: [ls[0]] + ls[2:], "cycles must start at 1"),
        (lambda ls: [ls[0], ls[0]] + ls[1:], "duplicate"),
        (lambda ls: [ls[0].replace(ls[0].split()[3], "abc", 1)] + ls[1:], "non-numeric"),
    ],
)
def test_schema_violations(synth_dir, tmp_path, mutate, check):  # TEST-DATA-02
    p = tmp_path / "bad.txt"
    p.write_text("\n".join(mutate(_lines(synth_dir))) + "\n")
    with pytest.raises(DataValidationError, match=check):
        D.read_sensor_file(p)


def test_checksum_tamper_deletes_file(tmp_path):  # TEST-DATA-04
    payload = b"1 1 0.0\n"
    manifest = {
        "source": {"repo": "r", "revision": "v", "url_template": "{repo}/{revision}/{file}"},
        "subsets": {"FD001": {"files": {"train_FD001.txt": "0" * 64}}},
    }

    def fake_fetch(url, dest):
        dest.write_bytes(payload)

    with pytest.raises(ChecksumError, match="checksum mismatch"):
        D.download(tmp_path, ("FD001",), manifest=manifest, fetch=fake_fetch)
    assert not (tmp_path / "FD001" / "train_FD001.txt").exists()


def test_download_verifies_and_writes_provenance(tmp_path):
    import hashlib

    payload = b"hello"
    sha = hashlib.sha256(payload).hexdigest()
    manifest = {
        "source": {"repo": "r", "revision": "v", "url_template": "{repo}/{revision}/{file}"},
        "subsets": {"FD001": {"files": {"x.txt": sha}}},
    }
    D.download(tmp_path, ("FD001",), manifest=manifest, fetch=lambda u, d: d.write_bytes(payload))
    assert D.verify(tmp_path, ("FD001",), manifest) == {"FD001/x.txt": True}
    assert (tmp_path / "PROVENANCE.json").exists()


def test_train_rul_labels():  # TEST-RUL-01/02
    df = pd.DataFrame({"unit": [1] * 5, "cycle": [1, 2, 3, 4, 5]})
    out = D.add_train_rul(df)
    assert out["rul"].tolist() == [4, 3, 2, 1, 0]
    assert D.cap_rul(out["rul"].to_numpy(), 3).tolist() == [3, 3, 2, 1, 0]
    assert out["rul"].tolist() == [4, 3, 2, 1, 0]  # uncapped column unchanged
    assert D.cap_rul(out["rul"].to_numpy(), None).tolist() == [4, 3, 2, 1, 0]


def test_dense_test_rul():  # TEST-RUL-03
    df = pd.DataFrame({"unit": [1, 1, 1], "cycle": [1, 2, 3]})
    assert D.dense_test_rul(df, np.array([10])).tolist() == [12, 11, 10]


def test_test_labels_map_to_units(synth_dir):  # TEST-RUL-04
    y = D.load_test_rul(synth_dir)
    assert y.tolist() == TEST_RUL  # label i <-> unit i+1 (file order)


def test_missing_file_message(tmp_path):
    with pytest.raises(DataValidationError, match="engine-rul data download"):
        D.load_train(tmp_path)


@requires_data
@pytest.mark.data
def test_real_fd001_invariants():  # TEST-DATA-03 / TEST-DATA-05
    man = D.load_manifest()
    assert all(D.verify("data/raw", ("FD001",)).values())
    inv = man["subsets"]["FD001"]["invariants"]
    train = D.load_train("data/raw")
    test = D.load_test_features("data/raw")
    D.check_train_invariants(train, inv)
    D.check_test_invariants(test, D.load_test_rul("data/raw"), inv)
    assert train.groupby("unit")["rul"].min().eq(0).all()
