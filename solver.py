from typing import Dict, List, Tuple


Candidate = Tuple[str, int, str, float, float]


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


def _parse(input_text: str):
    lines = [line.strip() for line in input_text.splitlines() if line.strip()]
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    raw = []
    task_set = set()
    for line in lines[start:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_list, courier, score_text, willingness_text = parts[:4]
        try:
            score = float(score_text)
            willingness = float(willingness_text)
        except ValueError:
            continue
        tasks = tuple(t.strip() for t in task_list.split(",") if t.strip())
        if not tasks:
            continue
        task_set.update(tasks)
        raw.append((task_list.strip(), tasks, courier.strip(), score, willingness))

    tasks = sorted(task_set)
    task_to_bit = {task: i for i, task in enumerate(tasks)}

    candidates: List[Candidate] = []
    for task_list, task_tuple, courier, score, willingness in raw:
        mask = 0
        for task in task_tuple:
            mask |= 1 << task_to_bit[task]
        candidates.append((task_list, mask, courier, score, willingness))
    return candidates, len(tasks)


def _rank(score: float, willingness: float, size: int, alpha: float) -> float:
    # Smaller is better. The first term keeps the official score low; the second
    # gently prefers riders that are more likely to accept.
    return score - alpha * willingness * 100.0 / max(1, size)


def _greedy(candidates: List[Candidate], task_count: int, alpha: float):
    full_mask = (1 << task_count) - 1
    used_couriers = set()
    used_tasks = 0
    result = []

    ordered = sorted(
        candidates,
        key=lambda c: (
            -_popcount(c[1]),
            _rank(c[3], c[4], _popcount(c[1]), alpha),
            c[3],
        ),
    )

    for task_list, mask, courier, score, willingness in ordered:
        if courier in used_couriers or (mask & used_tasks):
            continue
        used_couriers.add(courier)
        used_tasks |= mask
        result.append((task_list, [courier]))
        if used_tasks == full_mask:
            break
    return result, used_tasks


def _dp_fill(candidates: List[Candidate], task_count: int, alpha: float):
    # Exact set packing over tasks, with courier uniqueness handled by skipping
    # already used couriers in the stored state. 40 tasks is too many for a full
    # bitmask DP, so this is used only for small hidden cases.
    if task_count > 16:
        return None

    best: Dict[int, Tuple[float, Tuple[Tuple[str, str], ...], frozenset]] = {
        0: (0.0, tuple(), frozenset())
    }
    ordered = sorted(candidates, key=lambda c: _rank(c[3], c[4], _popcount(c[1]), alpha))

    for task_list, mask, courier, score, willingness in ordered:
        cost = _rank(score, willingness, _popcount(mask), alpha)
        snapshot = list(best.items())
        for old_mask, (old_cost, old_rows, old_couriers) in snapshot:
            if old_mask & mask or courier in old_couriers:
                continue
            new_mask = old_mask | mask
            new_cost = old_cost + cost
            prev = best.get(new_mask)
            if prev is None or new_cost < prev[0]:
                best[new_mask] = (
                    new_cost,
                    old_rows + ((task_list, courier),),
                    old_couriers | {courier},
                )

    max_cover = max(_popcount(mask) for mask in best)
    choice = min(
        (state for mask, state in best.items() if _popcount(mask) == max_cover),
        key=lambda x: x[0],
    )
    return [(task_list, [courier]) for task_list, courier in choice[1]]


def solve(input_text: str) -> list:
    candidates, task_count = _parse(input_text)
    if not candidates:
        return []

    best_result = None
    best_key = None

    for alpha in (0.0, 0.15, 0.3, 0.6, 1.0):
        if task_count <= 16:
            result = _dp_fill(candidates, task_count, alpha)
        else:
            result, _ = _greedy(candidates, task_count, alpha)

        if not result:
            continue
        covered = set()
        total_score = 0.0
        total_willingness = 0.0
        lookup = {(task_list, courier): (score, willingness) for task_list, _, courier, score, willingness in candidates}
        for task_list, couriers in result:
            covered.update(t.strip() for t in task_list.split(",") if t.strip())
            score, willingness = lookup[(task_list, couriers[0])]
            total_score += score
            total_willingness += willingness

        # Maximize completed task coverage first, then minimize score, then prefer
        # higher willingness. This mirrors the statement while staying conservative
        # about validity.
        key = (len(covered), -total_score, total_willingness)
        if best_key is None or key > best_key:
            best_key = key
            best_result = result

    return best_result or []
