"""`"skip": true` trong manifest được build_network và tune tôn trọng.

Chạy:  python3 -m pytest -q tests/test_skip.py
"""

import pytest

from pose_assoc.build_network import is_skipped, select_sequences
from pose_assoc.tune import classify, register_validation, seq_key, state_message

MANIFEST = {
    "validation": [{"name": "v1"}],
    "test": [{"name": "t1"}, {"name": "t2", "skip": True},
             {"name": "t3", "skip": False}, {"name": "t4", "skip": "true"}],
}


def test_is_skipped_needs_boolean_true():
    """Cùng quy ước với scripts/wham_gt_mapping.py: chỉ đúng `true` của JSON."""
    assert is_skipped({"skip": True})
    assert not is_skipped({})
    assert not is_skipped({"skip": False})
    assert not is_skipped({"skip": "true"})


def test_split_drops_skipped_sequences(capsys):
    names = [s["name"] for s in select_sequences(MANIFEST, None, "test")]
    assert names == ["t1", "t3", "t4"]
    assert "t2 === skip=true" in capsys.readouterr().out


def test_named_seq_runs_even_if_skipped(capsys):
    assert [s["name"] for s in select_sequences(MANIFEST, "t2", None)] == ["t2"]
    assert "vẫn chạy vì gọi bằng --seq" in capsys.readouterr().out


def test_unknown_seq_still_fails():
    with pytest.raises(SystemExit):
        select_sequences(MANIFEST, "khong-co", None)


def labeled_seq(tmp_path, name, **extra):
    out = tmp_path / name
    out.mkdir()
    (out / "mapping_frames.csv").write_text("frame,body_id,person\n0,1,S1\n")
    return {"name": name, "wham_pkl": name * 10, "wham_pkl_is_drive": True,
            "out_dir": out, "gt": {"path": tmp_path / "annot.mat"}, **extra}


def test_tune_does_not_add_skipped_sequence(tmp_path, capsys):
    seq = labeled_seq(tmp_path, "newseq", skip=True)
    reg = {"sequences": {}, "rounds": []}
    assert classify(seq, seq_key(seq), reg, set()) == "skip"
    # mod = None: nếu tune định dựng mạng cho sequence này thì sẽ lỗi ngay
    changed = register_validation(None, {"validation": [seq], "test": []}, reg, None, 1)
    assert changed == 0 and reg["sequences"] == {}
    assert "skip=true" in capsys.readouterr().out


def test_tune_skip_keeps_registered_data_and_points_to_exclude(tmp_path):
    seq = labeled_seq(tmp_path, "oldseq", skip=True)
    key = seq_key(seq)
    entry = {"name": "oldseq", "mapping_sig": "khac", "excluded": False, "added_round": 2}
    reg = {"sequences": {key: dict(entry)}, "rounds": []}
    # nhãn đã đổi nhưng vẫn không cập nhật vì skip
    assert classify(seq, key, reg, set()) == "skip"
    assert register_validation(None, {"validation": [seq], "test": []}, reg, None, 3) == 0
    assert reg["sequences"][key] == entry
    msg = state_message("skip", seq, entry)
    assert "lần 2" in msg and "--exclude oldseq" in msg
    assert "--exclude" not in state_message("skip", seq, dict(entry, excluded=True))


def test_test_overlap_still_rejected_when_skipped(tmp_path):
    seq = labeled_seq(tmp_path, "leak", skip=True)
    assert classify(seq, seq_key(seq), {"sequences": {}}, {seq_key(seq)}) == "test"


def test_unskipped_sequences_behave_as_before(tmp_path):
    seq = labeled_seq(tmp_path, "plain")
    reg = {"sequences": {}, "rounds": []}
    assert classify(seq, seq_key(seq), reg, set()) == "new"
    (seq["out_dir"] / "mapping_frames.csv").unlink()
    assert classify(seq, seq_key(seq), reg, set()) == "unlabeled"
