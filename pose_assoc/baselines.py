from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from pose_assoc.costs import MIN_VALID_JOINTS, displacement_2d
from pose_assoc.solvers.single_mcf import shortest_paths

# ω: bán kính cửa sổ majority vote của B5, tính bằng frame (cửa sổ dài 2ω + 1).
# Chọn trên validation cho từng bộ tham số (README mục 7): round 9 -> 20, round 7 -> 160.
# Mặc định theo round 9; bộ tham số khác thì truyền --vote-window của compare.
VOTE_WINDOW = 20

# B5' (B5 không dùng body_id): τ = ngưỡng độ dịch 2D để nối hai detection ở hai frame
# liên tiếp (cùng đơn vị với C_V: RMS độ dịch khớp / cỡ người, chia dt^p);
# ω' = bán kính cửa sổ vote dọc chuỗi đã nối. Chọn trên validation (README mục 7):
# round 9 -> τ = 1.0, ω' = 20; round 7 -> τ = 0.3, ω' = 320. Mặc định theo round 9.
LINK_THRESHOLD = 1.0
LINK_WINDOW = 20


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


def _vote_key(label):
    """Thứ tự cố định khi hoà phiếu: người trước None, rồi theo tên."""
    return (label is None, str(label))


def majority_vote_labels(labels: dict, window: int) -> dict:
    """
    Làm mượt nhãn của detection theo thời gian, dọc theo body_id của WHAM.

    labels: {(frame, body_id): người hoặc None}. Mỗi detection (t, d) lấy nhãn
    chiếm đa số trong các detection cùng body_id d ở frame t' với |t' - t| <= window,
    kể cả chính nó. None (không thuộc người GT nào) cũng là một nhãn được bỏ phiếu.
    Khoảng cách tính theo số frame, không theo số lần body_id xuất hiện.

    Hoà phiếu thì giữ nhãn cũ của (t, d) nếu nó nằm trong nhóm hoà; nếu không thì
    lấy theo _vote_key. window = 0 trả lại đúng labels.
    """
    window = int(window)
    if window < 0:
        raise ValueError(f"window phải >= 0, nhận {window}")

    frames_of = defaultdict(list)
    for frame, body_id in labels:
        frames_of[body_id].append(frame)

    smoothed = {}
    for body_id, frames in frames_of.items():
        frames.sort()
        counter = Counter()
        lo = hi = 0
        for frame in frames:
            while hi < len(frames) and frames[hi] <= frame + window:
                counter[labels[(frames[hi], body_id)]] += 1
                hi += 1
            while frames[lo] < frame - window:
                counter[labels[(frames[lo], body_id)]] -= 1
                lo += 1
            best = max(counter.values())
            tied = [label for label, count in counter.items() if count == best]
            own = labels[(frame, body_id)]
            smoothed[(frame, body_id)] = (
                own if own in tied else min(tied, key=_vote_key)
            )
    return smoothed


def _detections(net) -> list:
    """Mọi detection (frame, body_id) có nút trong mạng, không lặp."""
    seen = {}
    for node in net.nodes:
        if _get(node, "kind") in ("det", "detection"):
            seen[(int(_get(node, "frame")), _get(node, "body_id"))] = True
    return list(seen)


def link_distance(uv_a, uv_b) -> float:
    """Độ dịch 2D giữa hai detection cho bộ nối của B5' (như displacement_2d).

    NaN nếu thiếu toạ độ hoặc có ít hơn MIN_VALID_JOINTS khớp chung. displacement_2d
    trả 0 trong trường hợp đó (C_V coi là không có bằng chứng), còn ở đây 0 lại là
    "nối chắc chắn", nên phải tách ra.
    """
    if uv_a is None or uv_b is None:
        return float("nan")
    uv_a, uv_b = np.asarray(uv_a, float), np.asarray(uv_b, float)
    valid = np.isfinite(uv_a).all(1) & np.isfinite(uv_b).all(1)
    if valid.sum() < MIN_VALID_JOINTS:
        return float("nan")
    return displacement_2d(uv_a, uv_b)


def link_tracks_2d(detections, det_uv: dict, threshold: float = LINK_THRESHOLD,
                   gap_power: float = 1.0) -> dict:
    """
    Nối detection thành chuỗi chỉ bằng vị trí 2D, không dùng body_id của WHAM.

    detections: các (frame, body_id); body_id chỉ là khoá để tra toạ độ.
    det_uv: {(frame, body_id): (12, 2) px}, như load_positions().

    Mỗi frame có detection được ghép với frame có detection liền trước bằng Hungarian
    trên link_distance / dt^gap_power (cùng cách chia dt với C_V). Ghép tối đa số
    cặp hợp lệ, rồi tổng độ dịch nhỏ nhất; chỉ giữ cặp có giá trị <= threshold
    (threshold = inf: nối mọi cặp có toạ độ). Detection không được nối mở chuỗi mới.

    Trả {(frame, body_id): mã chuỗi}. Trong một frame mỗi chuỗi có tối đa một detection.
    """
    if not threshold >= 0:
        raise ValueError(f"threshold phải >= 0, nhận {threshold}")

    by_frame = defaultdict(list)
    for frame, body_id in detections:
        by_frame[int(frame)].append(body_id)

    track_of = {}
    next_track = 0
    prev_frame = None
    for frame in sorted(by_frame):
        cur = sorted(set(by_frame[frame]), key=str)
        linked = {}
        if prev_frame is not None:
            prev = sorted(set(by_frame[prev_frame]), key=str)
            gap = max(frame - prev_frame, 1)
            cost = np.full((len(prev), len(cur)), np.inf)
            for i, a in enumerate(prev):
                for j, b in enumerate(cur):
                    d = link_distance(det_uv.get((prev_frame, a)), det_uv.get((frame, b)))
                    if np.isfinite(d):
                        cost[i, j] = d / gap ** gap_power
            ok = np.isfinite(cost) & (cost <= threshold)
            if ok.any():
                # Cặp không hợp lệ mang chi phí rất lớn: Hungarian ghép nhiều cặp hợp lệ
                # nhất trước, rồi mới xét tổng độ dịch; cặp không hợp lệ bị bỏ sau đó.
                rows, cols = linear_sum_assignment(np.where(ok, cost, 1e9))
                for r, c in zip(rows, cols):
                    if ok[r, c]:
                        linked[cur[c]] = track_of[(prev_frame, prev[r])]
        for body_id in cur:
            if body_id in linked:
                track_of[(frame, body_id)] = linked[body_id]
            else:
                track_of[(frame, body_id)] = next_track
                next_track += 1
        prev_frame = frame
    return track_of


def _vote_along_tracks(net, track_of: dict, window: int):
    """
    B2 rồi majority vote nhãn dọc theo chuỗi detection (dùng chung cho B5 và B5').

    1. Chạy B2. Mỗi detection (t, d) nhận nhãn là người được gán nó, hoặc None.
    2. Làm mượt nhãn dọc theo track_of[(t, d)] trong cửa sổ ±window frame
       (majority_vote_labels).
    3. Ở mỗi (frame, người), chọn detection có nhãn mới là người đó; nếu nhiều thì
       lấy chi phí nút thấp nhất; nếu không có thì missed.

    Mỗi detection chỉ có một nhãn nên không bị hai người dùng chung.
    """
    base = hungarian_per_frame(net)
    groups, _, _, assignment = _empty_assignment(net)
    id_to_node = {_node_id(node, i): node for i, node in enumerate(net.nodes)}

    labels = {}
    for (frame, _person), group in groups.items():
        for _, node in group["det"]:
            labels.setdefault((frame, _get(node, "body_id")), None)
    for (frame, person), node_id in base.items():
        if node_id is not None:
            node = id_to_node[int(node_id)]
            labels[(frame, _get(node, "body_id"))] = person

    track_labels = {(frame, track_of[(frame, body_id)]): label
                    for (frame, body_id), label in labels.items()}
    if len(track_labels) != len(labels):
        raise ValueError("track_of gán hai detection của cùng một frame vào một chuỗi")
    smoothed_tracks = majority_vote_labels(track_labels, window)
    smoothed = {(frame, body_id): smoothed_tracks[(frame, track_of[(frame, body_id)])]
                for frame, body_id in labels}

    for (frame, person), group in groups.items():
        candidates = [
            item for item in group["det"]
            if smoothed.get((frame, _get(item[1], "body_id"))) == person
        ]
        if candidates:
            best = min(
                candidates,
                key=lambda item: (_det_cost(net, item), str(_get(item[1], "body_id"))),
            )
            assignment[(frame, person)] = _node_id(best[1], best[0])

    return assignment


def hungarian_majority_vote(net, window: int = VOTE_WINDOW):
    """
    B5 (A3 trong docs/MCMCF_revised.docx): B2 + majority vote dọc body_id của WHAM
    trong ±window frame (_vote_along_tracks).

    Khác B4, B5 dùng body_id của WHAM để nối detection giữa các frame (như B6), và
    làm theo phiếu bầu dù chi phí detection có cao hơn chi phí missed.
    """
    track_of = {key: key[1] for key in _detections(net)}
    return _vote_along_tracks(net, track_of, window)


def hungarian_vote_2d_tracks(net, det_uv: dict, window: int = LINK_WINDOW,
                             threshold: float = LINK_THRESHOLD, gap_power: float = 1.0):
    """
    B5': như B5 nhưng không dùng body_id của WHAM. Chuỗi detection được nối bằng
    độ dịch 2D (link_tracks_2d, cùng đại lượng với C_V của B4), rồi vote dọc chuỗi
    trong ±window frame.

    So với B4 ở round có eta > 0, B5' và B4 dùng cùng bằng chứng (PA-MPJPE, C_2D nếu
    có, độ dịch 2D); chỉ khác ở chỗ vote cục bộ thay cho tối ưu toàn cục.
    """
    track_of = link_tracks_2d(_detections(net), det_uv, threshold, gap_power)
    return _vote_along_tracks(net, track_of, window)


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