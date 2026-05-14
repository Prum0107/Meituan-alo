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
    for task_list, tasks, courier, score, willingness in candidates:
        lookup[(task_list, courier)] = (tasks, score, willingness)

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

    best_chosen = None
    best_score = None
    for order_key in strategies:
        chosen, used_couriers, used_tasks = build_base(order_key)
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

    return [(task_list, couriers) for task_list, couriers in best_chosen.items()]
