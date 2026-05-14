def solve(input_text: str) -> list:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    candidates = []
    task_set = set()
    for line in lines[start:]:
        parts = line.strip().split("\t")
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
        candidates.append((task_list.strip(), tasks, courier.strip(), score, willingness))

    if not candidates:
        return []

    all_tasks = set(task_set)
    lookup = {}
    group_rows = {}
    group_tasks = {}
    for task_list, tasks, courier, score, willingness in candidates:
        lookup[(task_list, courier)] = (tasks, score, willingness)
        group_rows.setdefault(task_list, []).append((task_list, tasks, courier, score, willingness))
        group_tasks[task_list] = tasks

    for task_list in group_rows:
        group_rows[task_list].sort(key=lambda c: (100.0 * len(c[1]) * (1.0 - c[4]) + c[3] * c[4], c[3]))

    def penalty(chosen):
        miss = dict((task, 1.0) for task in all_tasks)
        cost = 0.0
        for task_list, couriers in chosen.items():
            for courier in couriers:
                tasks, score, willingness = lookup[(task_list, courier)]
                cost += score * willingness
                for task in tasks:
                    miss[task] *= 1.0 - willingness
        return 100.0 * sum(miss.values()) + cost

    def build_base(order_key):
        used_couriers = set()
        used_tasks = set()
        chosen = {}

        for task_list, tasks, courier, score, willingness in sorted(candidates, key=order_key):
            if courier in used_couriers:
                continue
            if any(task in used_tasks for task in tasks):
                continue
            chosen[task_list] = [courier]
            used_couriers.add(courier)
            for task in tasks:
                used_tasks.add(task)
            if used_tasks == all_tasks:
                break

        return chosen, used_couriers, used_tasks

    def base_state(chosen):
        used_couriers = set()
        used_tasks = set()
        for task_list, couriers in chosen.items():
            used_couriers.update(couriers)
            used_tasks.update(group_tasks[task_list])
        return used_couriers, used_tasks

    def fill_uncovered(chosen):
        used_couriers, used_tasks = base_state(chosen)
        for task_list, tasks, courier, score, willingness in sorted(
            candidates,
            key=lambda c: (
                c[3] * c[4] + 100.0 * len(c[1]) * (1.0 - c[4]),
                -len(c[1]),
                c[3],
            ),
        ):
            if used_tasks == all_tasks:
                break
            if courier in used_couriers or task_list in chosen:
                continue
            if any(task in used_tasks for task in tasks):
                continue
            chosen[task_list] = [courier]
            used_couriers.add(courier)
            for task in tasks:
                used_tasks.add(task)
        return chosen

    def base_penalty(chosen):
        covered = set()
        for task_list in chosen:
            covered.update(group_tasks[task_list])
        return penalty(chosen) + 10000.0 * (len(all_tasks) - len(covered))

    def polish_base(chosen):
        # Try small replacements before adding backup couriers. This can turn two
        # weak single-order groups into one stronger bundled group and frees a
        # courier for later probability reinforcement.
        promising = sorted(
            candidates,
            key=lambda c: (
                c[3] * c[4] + 100.0 * len(c[1]) * (1.0 - c[4]),
                c[3] - 100.0 * len(c[1]) * c[4],
            ),
        )[:120]

        best = dict((k, list(v)) for k, v in chosen.items())
        best_score = base_penalty(best)
        for _ in range(1):
            changed = False
            used_couriers, _ = base_state(best)
            for task_list, tasks, courier, score, willingness in promising:
                conflicts = [
                    old_task_list
                    for old_task_list in best
                    if any(task in group_tasks[old_task_list] for task in tasks)
                ]
                if not conflicts:
                    continue
                freed = set()
                for old_task_list in conflicts:
                    freed.update(best[old_task_list])
                if courier in used_couriers and courier not in freed:
                    continue

                trial = dict((k, list(v)) for k, v in best.items() if k not in conflicts)
                trial[task_list] = [courier]
                fill_uncovered(trial)
                trial_score = base_penalty(trial)
                if trial_score + 1e-9 < best_score:
                    best = trial
                    best_score = trial_score
                    changed = True
                    break
            if not changed:
                break
        return best

    def build_group_base(group_key):
        used_couriers = set()
        used_tasks = set()
        chosen = {}

        for task_list in sorted(group_rows, key=group_key):
            tasks = group_tasks[task_list]
            if any(task in used_tasks for task in tasks):
                continue

            best = None
            best_value = None
            for row in group_rows[task_list]:
                _, _, courier, score, willingness = row
                if courier in used_couriers:
                    continue
                value = 100.0 * len(tasks) * (1.0 - willingness) + score * willingness
                if best_value is None or value < best_value:
                    best_value = value
                    best = row

            if best is None:
                continue

            _, _, courier, score, willingness = best
            chosen[task_list] = [courier]
            used_couriers.add(courier)
            for task in tasks:
                used_tasks.add(task)
            if used_tasks == all_tasks:
                break

        return chosen, used_couriers, used_tasks

    def improve(chosen, used_couriers, used_tasks):
        group_miss = {}
        for task_list, couriers in chosen.items():
            miss = 1.0
            for courier in couriers:
                tasks, score, willingness = lookup[(task_list, courier)]
                miss *= 1.0 - willingness
            group_miss[task_list] = miss

        # First repair coverage if a greedy base left any task uncovered.
        changed = True
        while changed:
            changed = False
            best = None
            best_new = 0
            for task_list, tasks, courier, score, willingness in candidates:
                if courier in used_couriers or task_list in chosen:
                    continue
                if any(task in used_tasks for task in tasks):
                    continue
                new_count = sum(1 for task in tasks if task not in used_tasks)
                if new_count > best_new:
                    best_new = new_count
                    best = (task_list, tasks, courier, score, willingness)
            if best is not None:
                task_list, tasks, courier, score, willingness = best
                chosen[task_list] = [courier]
                used_couriers.add(courier)
                group_miss[task_list] = 1.0 - willingness
                for task in tasks:
                    used_tasks.add(task)
                changed = True

        # Then add extra couriers to already selected task groups while the
        # marginal expected-penalty reduction is positive.
        while True:
            best = None
            best_gain = 0.0
            for task_list, tasks, courier, score, willingness in candidates:
                if courier in used_couriers or task_list not in chosen:
                    continue
                miss = group_miss[task_list]
                gain = 100.0 * len(tasks) * miss * willingness - score * willingness
                if gain > best_gain:
                    best_gain = gain
                    best = (task_list, courier, willingness)
            if best is None:
                break
            task_list, courier, willingness = best
            chosen[task_list].append(courier)
            used_couriers.add(courier)
            group_miss[task_list] *= 1.0 - willingness

    strategies = [
        lambda c: (-len(c[1]), c[3] * c[4] + 100.0 * len(c[1]) * (1.0 - c[4]), c[3]),
        lambda c: (-len(c[1]), c[3], -c[4]),
        lambda c: (c[3] * c[4] + 100.0 * len(c[1]) * (1.0 - c[4]), -len(c[1]), c[3]),
        lambda c: (c[3], -len(c[1]), -c[4]),
        lambda c: (c[3] - 100.0 * len(c[1]) * c[4], c[3]),
    ]
    group_strategies = [
        lambda g: (-len(group_tasks[g]), group_rows[g][0][3] * group_rows[g][0][4] + 100.0 * len(group_tasks[g]) * (1.0 - group_rows[g][0][4])),
        lambda g: (-len(group_tasks[g]), group_rows[g][0][3]),
        lambda g: (group_rows[g][0][3] * group_rows[g][0][4] + 100.0 * len(group_tasks[g]) * (1.0 - group_rows[g][0][4]), -len(group_tasks[g])),
        lambda g: (group_rows[g][0][3] - 100.0 * len(group_tasks[g]) * group_rows[g][0][4], group_rows[g][0][3]),
    ]

    best_chosen = None
    best_score = None
    for order_key in strategies:
        chosen, used_couriers, used_tasks = build_base(order_key)
        chosen = polish_base(chosen)
        used_couriers, used_tasks = base_state(chosen)
        improve(chosen, used_couriers, used_tasks)
        score = penalty(chosen)
        covered = set()
        for task_list in chosen:
            covered.update(lookup[(task_list, chosen[task_list][0])][0])
        # Prefer full coverage strongly, then lower expected penalty.
        score += 10000.0 * (len(all_tasks) - len(covered))
        if best_score is None or score < best_score:
            best_score = score
            best_chosen = chosen

    for group_key in group_strategies:
        chosen, used_couriers, used_tasks = build_group_base(group_key)
        chosen = polish_base(chosen)
        used_couriers, used_tasks = base_state(chosen)
        improve(chosen, used_couriers, used_tasks)
        score = penalty(chosen)
        covered = set()
        for task_list in chosen:
            covered.update(lookup[(task_list, chosen[task_list][0])][0])
        score += 10000.0 * (len(all_tasks) - len(covered))
        if best_score is None or score < best_score:
            best_score = score
            best_chosen = chosen

    return [(task_list, couriers) for task_list, couriers in best_chosen.items()]
