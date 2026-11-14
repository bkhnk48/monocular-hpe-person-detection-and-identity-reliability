from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from pose_assoc.solvers.single_mcf import shortest_paths


def _get(obj: Any, key: str, default=None):
    """Đọc field từ dict hoặc dataclass/object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _node_id(node: Any, index: int) -> int:
    value = _get(node, "id", index)
    return int(value)


def _node_cost(net, node: Any, index: int) -> float:
    """Chi phí unary của node, dùng đúng chi phí trong Network."""
    costs = np.asarray(net.node_costs, dtype=float)
    return float(costs[_node_id(node, index)])


def _nodes_by_frame_person(net):
    """
    Trả về:
      groups[(frame, person)] = {"det": [...], "miss": node}

    Mỗi phần tử detection là (node_index, node).
    """
    groups = defaultdict(lambda: {"det": [], "miss": None})

    for i, node in enumerate(net.nodes):
        kind = _get(node, "kind")
        frame = _get(node, "frame")
        person = _get(node, "person")

        if frame is None or person is None:
            continue

        key = (int(frame), person)

        if kind in ("det", "detection"):
            groups[key]["det"].append((i, node))
        elif kind == "miss":
            groups[key]["miss"] = (i, node)

    return groups


def _frames_and_persons(net):
    groups = _nodes_by_frame_person(net)
    frames = sorted({frame for frame, _ in groups})
    persons = list(net.persons)
    return groups, frames, persons


def _miss_cost(net, group) -> float:
    item = group["miss"]
    if item is None:
        raise ValueError("Không tìm thấy miss node cho một (frame, person).")
    i, node = item
    return _node_cost(net, node, i)


def _det_cost(net, item) -> float:
    i, node = item
    return _node_cost(net, node, i)


def _empty_assignment(net):
    groups, frames, persons = _frames_and_persons(net)
    assignment = {}

    for frame in frames:
        for person in persons:
            if (frame, person) in groups:
                assignment[(frame, person)] = None

    return groups, frames, persons, assignment


def greedy_per_frame(net):
    """
    B1: Greedy assignment độc lập theo từng frame.

    Trong mỗi frame:
      1. Tạo các cặp (person, detection).
      2. Tính lợi ích so với miss:
             benefit = miss_cost - detection_cost
      3. Sắp xếp lợi ích giảm dần.
      4. Chọn cặp có benefit > 0, không dùng lại person/detection.

    Kết quả:
      assignment[(frame, person)] = node_id hoặc None
    """
    groups, frames, persons, assignment = _empty_assignment(net)

    for frame in frames:
        candidates = []

        for person in persons:
            group = groups.get((frame, person))
            if group is None:
                continue

            miss_cost = _miss_cost(net, group)

            for item in group["det"]:
                det_cost = _det_cost(net, item)
                benefit = miss_cost - det_cost

                if benefit > 0:
                    node_index, node = item
                    candidates.append(
                        (
                            -benefit,
                            det_cost,
                            str(person),
                            int(_get(node, "body_id")),
                            person,
                            node_index,
                        )
                    )

        candidates.sort()

        used_persons = set()
        used_detections = set()

        for (
            _negative_benefit,
            _cost,
            _person_key,
            _body_id,
            person,
            node_index,
        ) in candidates:
            node = net.nodes[node_index]
            detection_key = (
                frame,
                _get(node, "body_id"),
            )

            if person in used_persons:
                continue
            if detection_key in used_detections:
                continue

            assignment[(frame, person)] = _node_id(node, node_index)
            used_persons.add(person)
            used_detections.add(detection_key)

    return assignment


def hungarian_per_frame(net):
    """
    B2: Hungarian assignment độc lập theo từng frame.

    Mỗi person có:
      - các cột detection dùng chung;
      - một cột dummy riêng để biểu diễn miss.

    Detection được chọn tối đa một lần trong một frame.
    Chi phí miss lấy trực tiếp từ miss node trong Network.
    """
    groups, frames, persons, assignment = _empty_assignment(net)

    for frame in frames:
        active_persons = [
            p for p in persons
            if (frame, p) in groups
        ]

        if not active_persons:
            continue

        # Tập detection duy nhất của frame.
        detections = {}
        for person in active_persons:
            for item in groups[(frame, person)]["det"]:
                _, node = item
                body_id = _get(node, "body_id")
                detections[body_id] = node

        body_ids = sorted(detections, key=lambda x: str(x))

        n_persons = len(active_persons)
        n_dets = len(body_ids)

        # Mỗi person có một dummy miss riêng.
        n_cols = n_dets + n_persons
        big_m = 1e9
        cost_matrix = np.full((n_persons, n_cols), big_m, dtype=float)

        # Map (person, body_id) -> node id.
        node_lookup = {}

        for row, person in enumerate(active_persons):
            group = groups[(frame, person)]

            for col, body_id in enumerate(body_ids):
                matching = [
                    item for item in group["det"]
                    if _get(item[1], "body_id") == body_id
                ]

                if matching:
                    # Thông thường chỉ có một node cho một cặp.
                    item = min(matching, key=lambda x: _det_cost(net, x))
                    cost_matrix[row, col] = _det_cost(net, item)
                    node_lookup[(person, body_id)] = _node_id(
                        item[1], item[0]
                    )

            # Chỉ cho phép person này dùng dummy của chính mình.
            miss_col = n_dets + row
            cost_matrix[row, miss_col] = _miss_cost(net, group)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for row, col in zip(row_ind, col_ind):
            person = active_persons[row]

            if col < n_dets:
                body_id = body_ids[col]
                node_id = node_lookup.get((person, body_id))

                if node_id is not None and cost_matrix[row, col] < big_m:
                    assignment[(frame, person)] = node_id

    return assignment


def independent_dp(net):
    """
    B3: Giải shortest path độc lập cho từng person.

    Không áp đặt ràng buộc shared-detection giữa các person.
    Vì vậy hai person có thể cùng chọn một detection.
    """
    flow = shortest_paths(net)
    return assignment_from_arc_flow(net, flow)


def assignment_from_arc_flow(net, flow):
    """
    Chuyển nghiệm flow trên arc thành assignment (frame, person) -> node_id.

    Một node thuộc đường đi nếu có ít nhất một arc đi vào node mang flow > 0.
    Source/sink không được đưa vào kết quả.
    """
    flow = np.asarray(flow, dtype=float)

    if flow.ndim != 1 or len(flow) != len(net.tails):
        raise ValueError(
            "flow phải là vector có độ dài bằng số arc trong Network."
        )

    incoming = np.zeros(len(net.nodes), dtype=float)

    for arc_index, value in enumerate(flow):
        if value > 0.5:
            head = int(net.heads[arc_index])
            if 0 <= head < len(incoming):
                incoming[head] += value

    assignment = {}

    for i, node in enumerate(net.nodes):
        kind = _get(node, "kind")
        frame = _get(node, "frame")
        person = _get(node, "person")

        if kind not in ("det", "detection", "miss"):
            continue
        if frame is None or person is None:
            continue
        if incoming[i] <= 0.5:
            continue

        key = (int(frame), person)

        if kind == "miss":
            assignment[key] = None
        else:
            assignment[key] = _node_id(node, i)

    return assignment


def original_wham_id_majority(net):
    """
    B6: Baseline WHAM ID nguyên bản.

    Bước 1:
      Với mỗi detection (frame, body_id), tìm person có PA-MPJPE nhỏ nhất.
      Detection này bỏ phiếu cho person đó.

    Bước 2:
      Mỗi body_id được gán cố định cho person có nhiều phiếu nhất.

    Bước 3:
      Ở từng frame, mỗi person chọn detection có body_id được gán cho mình
      và có PA-MPJPE thấp nhất.

    Nếu nhiều body_id cùng được majority-vote cho một person, bước 3
    vẫn chỉ chọn tối đa một detection cho person/frame.
    """
    groups, frames, persons, assignment = _empty_assignment(net)

    # Tập các detection node được biểu diễn theo (frame, body_id).
    detection_person_costs = defaultdict(dict)

    for (frame, person), group in groups.items():
        for item in group["det"]:
            _, node = item
            body_id = _get(node, "body_id")
            pa_mm = _get(node, "pa_mm")

            if pa_mm is None or not np.isfinite(float(pa_mm)):
                continue

            key = (frame, body_id)
            old = detection_person_costs[key].get(person)

            if old is None or float(pa_mm) < old[0]:
                detection_person_costs[key][person] = (
                    float(pa_mm),
                    item,
                )

    # Mỗi body_id bỏ một phiếu cho person có PA-MPJPE thấp nhất
    # trong từng frame mà detection xuất hiện.
    votes = defaultdict(Counter)

    for (frame, body_id), person_costs in detection_person_costs.items():
        if not person_costs:
            continue

        winner = min(
            person_costs,
            key=lambda p: (person_costs[p][0], str(p)),
        )
        votes[body_id][winner] += 1

    # Ánh xạ cố định body_id -> person.
    body_to_person = {}

    for body_id, counter in votes.items():
        body_to_person[body_id] = min(
            counter,
            key=lambda p: (-counter[p], str(p)),
        )

    # Dùng ánh xạ cố định để chọn detection tại từng frame.
    for frame in frames:
        for person in persons:
            group = groups.get((frame, person))
            if group is None:
                continue

            candidates = []

            for item in group["det"]:
                _, node = item
                body_id = _get(node, "body_id")

                if body_to_person.get(body_id) != person:
                    continue

                pa_mm = _get(node, "pa_mm")
                if pa_mm is None or not np.isfinite(float(pa_mm)):
                    continue

                candidates.append(
                    (float(pa_mm), str(body_id), item)
                )

            if candidates:
                _, _, best_item = min(candidates)
                assignment[(frame, person)] = _node_id(
                    best_item[1], best_item[0]
                )

    return assignment, body_to_person