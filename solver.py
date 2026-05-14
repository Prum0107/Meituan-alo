def solve(input_text: str) -> list:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    candidates = []
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
        candidates.append((task_list.strip(), tasks, courier.strip(), score, willingness))

    # Keep only candidates that can improve the expected objective by themselves.
    # The inferred penalty is approximately:
    #   100 * expected_unaccepted_orders + sum(score * accept_probability)
    useful = []
    for task_list, tasks, courier, score, willingness in candidates:
        gain = 100.0 * len(tasks) * willingness - score * willingness
        if gain > 0:
            useful.append((task_list, tasks, courier, score, willingness))

    used_couriers = set()
    used_tasks = set()
    chosen = {}
    group_miss = {}
    group_tasks = {}

    while True:
        best = None
        best_gain = 0.0

        for task_list, tasks, courier, score, willingness in useful:
            if courier in used_couriers:
                continue

            if task_list in chosen:
                miss = group_miss[task_list]
            else:
                if any(task in used_tasks for task in tasks):
                    continue
                miss = 1.0

            gain = 100.0 * len(tasks) * miss * willingness - score * willingness
            if gain > best_gain:
                best_gain = gain
                best = (task_list, tasks, courier, score, willingness)

        if best is None:
            break

        task_list, tasks, courier, score, willingness = best
        if task_list not in chosen:
            chosen[task_list] = []
            group_miss[task_list] = 1.0
            group_tasks[task_list] = tasks
            for task in tasks:
                used_tasks.add(task)

        chosen[task_list].append(courier)
        used_couriers.add(courier)
        group_miss[task_list] *= 1.0 - willingness

    # If an unusual case still has uncovered orders, fill them conservatively
    # with the lowest-score non-conflicting candidates so the completion metric
    # does not collapse.
    remaining = sorted(candidates, key=lambda x: (len(x[1]), x[3]))
    for task_list, tasks, courier, score, willingness in remaining:
        if courier in used_couriers:
            continue
        if any(task in used_tasks for task in tasks):
            continue
        chosen[task_list] = [courier]
        used_couriers.add(courier)
        for task in tasks:
            used_tasks.add(task)

    return [(task_list, couriers) for task_list, couriers in chosen.items()]
