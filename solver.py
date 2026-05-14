import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    task_list: str
    tasks: Tuple[str, ...]
    mask: int
    courier: str
    score: float
    willingness: float


def _parse(input_text: str) -> Tuple[List[Candidate], List[str], List[str]]:
    lines = [line.strip() for line in input_text.splitlines() if line.strip()]
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    raw_rows = []
    tasks_seen = set()
    couriers_seen = set()
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
        raw_rows.append((task_list.strip(), tasks, courier.strip(), score, willingness))
        tasks_seen.update(tasks)
        couriers_seen.add(courier.strip())

    task_order = sorted(tasks_seen)
    courier_order = sorted(couriers_seen)
    task_to_bit = {task: i for i, task in enumerate(task_order)}

    candidates = []
    for task_list, tasks, courier, score, willingness in raw_rows:
        mask = 0
        for task in tasks:
            mask |= 1 << task_to_bit[task]
        candidates.append(Candidate(task_list, tasks, mask, courier, score, willingness))

    return candidates, task_order, courier_order


def _objective(
    selected: Sequence[Candidate],
    task_count: int,
    accept_weight: float,
    score_weight: float,
) -> float:
    miss_prob = [1.0] * task_count
    expected_score = 0.0
    for cand in selected:
        w = max(0.0, min(1.0, cand.willingness))
        expected_score += cand.score * w
        mask = cand.mask
        bit = 0
        while mask:
            if mask & 1:
                miss_prob[bit] *= 1.0 - w
            mask >>= 1
            bit += 1

    expected_accepted = sum(1.0 - p for p in miss_prob)
    return accept_weight * expected_accepted - score_weight * expected_score


def _marginal_gain(
    cand: Candidate,
    miss_prob: Sequence[float],
    accept_weight: float,
    score_weight: float,
) -> float:
    w = max(0.0, min(1.0, cand.willingness))
    gain = 0.0
    mask = cand.mask
    bit = 0
    while mask:
        if mask & 1:
            gain += miss_prob[bit] * w
        mask >>= 1
        bit += 1
    return accept_weight * gain - score_weight * cand.score * w


def _apply(cand: Candidate, miss_prob: List[float]) -> None:
    w = max(0.0, min(1.0, cand.willingness))
    mask = cand.mask
    bit = 0
    while mask:
        if mask & 1:
            miss_prob[bit] *= 1.0 - w
        mask >>= 1
        bit += 1


def _top_candidates_by_courier(
    candidates: Iterable[Candidate],
    keep_per_courier: int,
    blend: float,
) -> Dict[str, List[Candidate]]:
    by_courier: Dict[str, List[Candidate]] = {}
    for cand in candidates:
        by_courier.setdefault(cand.courier, []).append(cand)

    for courier, rows in by_courier.items():
        rows.sort(
            key=lambda c: (
                -len(c.tasks) * c.willingness * blend
                + c.score * c.willingness * (1.0 - blend) / 100.0,
                c.score,
            )
        )
        by_courier[courier] = rows[:keep_per_courier]
    return by_courier


def _greedy_build(
    by_courier: Dict[str, List[Candidate]],
    task_count: int,
    accept_weight: float,
    score_weight: float,
    rng: random.Random,
    noise: float,
) -> List[Candidate]:
    miss_prob = [1.0] * task_count
    available = set(by_courier)
    selected: List[Candidate] = []

    while available:
        best_cand: Optional[Candidate] = None
        best_gain = 0.0
        for courier in list(available):
            for cand in by_courier[courier]:
                gain = _marginal_gain(cand, miss_prob, accept_weight, score_weight)
                if noise:
                    gain *= rng.uniform(1.0 - noise, 1.0 + noise)
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand

        if best_cand is None:
            break
        selected.append(best_cand)
        available.remove(best_cand.courier)
        _apply(best_cand, miss_prob)

    return selected


def _improve_by_replacement(
    selected: List[Candidate],
    by_courier: Dict[str, List[Candidate]],
    task_count: int,
    accept_weight: float,
    score_weight: float,
    deadline: float,
) -> List[Candidate]:
    current = list(selected)
    current_by_courier = {cand.courier: i for i, cand in enumerate(current)}
    current_score = _objective(current, task_count, accept_weight, score_weight)

    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        for courier, options in by_courier.items():
            if time.perf_counter() >= deadline:
                break
            pos = current_by_courier.get(courier)
            old = current[pos] if pos is not None else None
            for cand in options:
                if old == cand:
                    continue
                trial = list(current)
                if pos is None:
                    trial.append(cand)
                else:
                    trial[pos] = cand
                trial_score = _objective(trial, task_count, accept_weight, score_weight)
                if trial_score > current_score + 1e-9:
                    current = trial
                    current_by_courier[courier] = len(current) - 1 if pos is None else pos
                    current_score = trial_score
                    improved = True
                    break
    return current


def _format(selected: Sequence[Candidate]) -> list:
    grouped: Dict[str, List[str]] = {}
    order: List[str] = []
    for cand in selected:
        if cand.task_list not in grouped:
            grouped[cand.task_list] = []
            order.append(cand.task_list)
        grouped[cand.task_list].append(cand.courier)
    return [(task_list, grouped[task_list]) for task_list in order]


def solve(input_text: str) -> list:
    candidates, task_order, _ = _parse(input_text)
    if not candidates:
        return []

    start = time.perf_counter()
    deadline = start + 9.0
    task_count = len(task_order)
    rng = random.Random(301)

    best: List[Candidate] = []
    best_score = -math.inf

    # The official statement prioritizes accepted order count and then total score.
    # These parameter sweeps let the agent explore that tradeoff without depending
    # on one hand-tuned heuristic.
    settings = [
        (12000.0, 1.00, 50, 0.00, 0.20),
        (10000.0, 0.80, 60, 0.00, 0.45),
        (9000.0, 0.55, 80, 0.03, 0.65),
        (8000.0, 0.35, 100, 0.05, 0.80),
        (7000.0, 0.20, 120, 0.08, 0.95),
    ]

    while time.perf_counter() < deadline:
        for accept_weight, score_weight, keep, noise, blend in settings:
            if time.perf_counter() >= deadline:
                break
            by_courier = _top_candidates_by_courier(candidates, keep, blend)
            selected = _greedy_build(
                by_courier,
                task_count,
                accept_weight,
                score_weight,
                rng,
                noise,
            )
            selected = _improve_by_replacement(
                selected,
                by_courier,
                task_count,
                accept_weight,
                score_weight,
                min(deadline, time.perf_counter() + 0.7),
            )
            score = _objective(selected, task_count, accept_weight, score_weight)
            if score > best_score:
                best_score = score
                best = selected

        # After one deterministic pass, keep doing noisy passes until the time budget
        # is nearly spent.
        settings = [
            (
                rng.choice([7000.0, 8500.0, 10000.0, 12000.0]),
                rng.choice([0.15, 0.35, 0.6, 0.9, 1.2]),
                rng.choice([50, 70, 90, 120, 160]),
                rng.uniform(0.02, 0.12),
                rng.uniform(0.2, 1.0),
            )
            for _ in range(4)
        ]

    return _format(best)


if __name__ == "__main__":
    import json
    import sys

    text = sys.stdin.read()
    print(json.dumps(solve(text), ensure_ascii=False))
