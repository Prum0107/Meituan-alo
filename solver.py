def solve(input_text: str) -> list:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    candidates = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue
        candidates.append(
            (score, -willingness, task_id_list_str.strip(), courier_id.strip())
        )

    candidates.sort()

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for score, neg_willingness, task_id_list_str, courier_id in candidates:
        task_ids = [t.strip() for t in task_id_list_str.split(",")]
        if courier_id in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in task_ids):
            continue

        assigned_couriers.add(courier_id)
        for task_id in task_ids:
            assigned_tasks.add(task_id)
        result.append((task_id_list_str, [courier_id]))

    return result
