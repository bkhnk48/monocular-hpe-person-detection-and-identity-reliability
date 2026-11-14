"""Hai định dạng nhãn trong pose_assoc/evaluation.py không được dùng lẫn.

Chạy:  python3 -m pytest -q tests/test_evaluation_truth.py

  load_truth()            {(frame, body_id): người hoặc None}  -> pose_assoc.compare
  load_truth_by_person()  {(frame, người): {body_id}}          -> build_network, tune

Nếu đưa nhãn của load_truth() vào evaluate() hay node_status(), mọi lần tra theo
(frame, người) đều hụt và mọi detection bị chấm wrong_id mà không có lỗi; test này
giữ cho chỗ đó báo lỗi rõ ràng.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pose_assoc.evaluation import evaluate, load_truth, load_truth_by_person
from pose_assoc.tune import node_status

ROOT = Path(__file__).resolve().parents[1]
CSV = "frame,body_id,person\n0,1,A\n0,2,B\n1,1,A\n1,3,\n"


@pytest.fixture
def mapping(tmp_path):
    path = tmp_path / "mapping_frames.csv"
    path.write_text(CSV)
    return path


def test_two_readers_give_their_own_format(mapping):
    assert load_truth_by_person(mapping) == {(0, "A"): {1}, (0, "B"): {2}, (1, "A"): {1}}
    assert load_truth(mapping_csv=str(mapping)) == {
        (0, 1): "A", (0, 2): "B", (1, 1): "A", (1, 3): None}


def test_evaluate_scores_each_status(mapping):
    assign = pd.DataFrame({"frame": [0, 0, 1, 1], "person": ["A", "B", "A", "B"],
                           "body_id": pd.array([1, None, 3, None], dtype="Int64")})
    df, m = evaluate(assign, load_truth_by_person(mapping))
    assert list(df["status"]) == ["correct", "missed", "wrong_id", "correct_miss"]
    assert m["accuracy"] == 0.5 and m["wrong_id"] == 1 and m["missed"] == 1


def test_wrong_format_is_rejected(mapping):
    flat = load_truth(mapping_csv=str(mapping))
    assign = pd.DataFrame({"frame": [0], "person": ["A"],
                           "body_id": pd.array([1], dtype="Int64")})
    with pytest.raises(TypeError, match="load_truth_by_person"):
        evaluate(assign, flat)
    with pytest.raises(TypeError, match="load_truth_by_person"):
        node_status(SimpleNamespace(nodes=[]), flat)


@pytest.mark.parametrize("seq", ["mpi_S8_Seq1_cam0", "h4d_041_grappling2_cam07"])
def test_readers_agree_on_committed_labels(seq):
    path = ROOT / "outputs" / "mapping" / seq / "mapping_frames.csv"
    if not path.is_file():
        pytest.skip(f"không có {path}")
    by_person = load_truth_by_person(path)
    flat = load_truth(mapping_csv=str(path))
    from_flat: dict = {}
    for (frame, body_id), person in flat.items():
        if person is not None:
            from_flat.setdefault((frame, person), set()).add(int(body_id))
    assert from_flat == by_person
