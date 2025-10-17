from typing import List, Tuple, Optional, Dict
import numpy as np

# Representation: assignment[i] in {-1, 0, 1, ..., K-1}
# -1 means the item is not placed in any knapsack


def get_problem_shape(values: np.ndarray, weights: np.ndarray, constraints: np.ndarray) -> Tuple[int, int, int]:
    num_items = int(values.shape[0])
    num_dimensions = int(weights.shape[1])
    num_knapsacks = int(constraints.shape[0])
    return num_knapsacks, num_items, num_dimensions


def compute_loads(assignment: np.ndarray, weights: np.ndarray, num_knapsacks: int) -> np.ndarray:
    num_dimensions = weights.shape[1]
    loads = np.zeros((num_knapsacks, num_dimensions), dtype=np.int64)
    for item_idx, k in enumerate(assignment):
        if k >= 0:
            loads[k] += weights[item_idx]
    return loads


def is_feasible_assignment(assignment: np.ndarray, weights: np.ndarray, constraints: np.ndarray) -> bool:
    num_knapsacks = constraints.shape[0]
    loads = compute_loads(assignment, weights, num_knapsacks)
    return np.all(loads <= constraints)


def objective_value(assignment: np.ndarray, values: np.ndarray) -> np.int64:
    return np.asarray(values[assignment >= 0], dtype=np.int64).sum()


def greedy_initialize(values: np.ndarray, weights: np.ndarray, constraints: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Greedy constructive heuristic (simple).
    Sort by value density and place into feasible knapsack with most slack.
    """
    K, N, _D = get_problem_shape(values, weights, constraints)

    densities = values / (1.0 + weights.sum(axis=1))
    item_order = np.argsort(-densities, kind="stable")

    assignment = np.full(N, -1, dtype=np.int64)
    remaining = constraints.astype(np.int64).copy()

    for i in item_order:
        slack = remaining.sum(axis=1)
        knapsack_order = np.argsort(-slack, kind="stable")
        for k in knapsack_order:
            if np.all(remaining[k] - weights[i] >= 0):
                assignment[i] = int(k)
                remaining[k] -= weights[i]
                break

    return assignment


def _can_move_item(
    item_idx: int,
    target_k: int,
    assignment: np.ndarray,
    loads: np.ndarray,
    weights: np.ndarray,
    constraints: np.ndarray,
) -> bool:
    current_k = int(assignment[item_idx])
    if target_k == current_k:
        return False

    if current_k >= 0:
        if np.any(loads[current_k] - weights[item_idx] < 0):
            return False

    if target_k >= 0:
        if np.any(loads[target_k] + weights[item_idx] > constraints[target_k]):
            return False

    return True


def _apply_move(
    item_idx: int,
    target_k: int,
    assignment: np.ndarray,
    loads: np.ndarray,
    weights: np.ndarray,
) -> None:
    current_k = int(assignment[item_idx])
    if current_k >= 0:
        loads[current_k] -= weights[item_idx]
    if target_k >= 0:
        loads[target_k] += weights[item_idx]
    assignment[item_idx] = int(target_k)


def best_improving_neighbor(
    assignment: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    constraints: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[Optional[Tuple[int, int]], np.int64]:
    """First-improvement search over single-item moves (including drop to -1)."""
    K, N, _D = get_problem_shape(values, weights, constraints)
    loads = compute_loads(assignment, weights, K)

    item_indices = np.arange(N)
    rng.shuffle(item_indices)

    for i in item_indices:
        current_k = int(assignment[i])
        targets = list(range(-1, K))
        rng.shuffle(targets)
        for target_k in targets:
            if target_k == current_k:
                continue
            if not _can_move_item(i, target_k, assignment, loads, weights, constraints):
                continue
            gain = np.int64(0)
            if current_k < 0 and target_k >= 0:
                gain = np.int64(values[i])
            elif current_k >= 0 and target_k < 0:
                gain = -np.int64(values[i])
            else:
                gain = np.int64(0)
            if gain > 0:
                return (i, target_k), gain

    return None, np.int64(0)


def hill_climbing(
    values: np.ndarray,
    weights: np.ndarray,
    constraints: np.ndarray,
    rng: np.random.Generator,
    max_iterations: int = 10000,
) -> Tuple[np.ndarray, np.int64, List[np.int64]]:
    """Naive hill climbing: greedy init + first-improvement single-item moves."""
    K, N, _D = get_problem_shape(values, weights, constraints)

    assignment = greedy_initialize(values, weights, constraints, rng)

    if not is_feasible_assignment(assignment, weights, constraints):
        densities = values / (1.0 + weights.sum(axis=1))
        for i in np.argsort(densities):
            if is_feasible_assignment(assignment, weights, constraints):
                break
            if assignment[i] >= 0:
                assignment[i] = -1

    trajectory: List[np.int64] = [objective_value(assignment, values)]

    iterations = 0
    while iterations < max_iterations:
        move, delta = best_improving_neighbor(assignment, values, weights, constraints, rng)
        if move is None or delta <= 0:
            break
        loads = compute_loads(assignment, weights, K)
        _apply_move(move[0], move[1], assignment, loads, weights)
        trajectory.append(trajectory[-1] + delta)
        iterations += 1

    return assignment, trajectory[-1], trajectory


def tabu_search(
    values: np.ndarray,
    weights: np.ndarray,
    constraints: np.ndarray,
    rng: np.random.Generator,
    max_iterations: int = 10000,
    tabu_tenure: int = 10,
    candidate_items: int = 200,
) -> Tuple[np.ndarray, np.int64, List[np.int64]]:
    """Tabu Search with single-item moves, aspiration, and simple tenure."""
    K, N, _D = get_problem_shape(values, weights, constraints)

    current_assignment = greedy_initialize(values, weights, constraints, rng)
    if not is_feasible_assignment(current_assignment, weights, constraints):
        densities = values / (1.0 + weights.sum(axis=1))
        for i in np.argsort(densities):
            if is_feasible_assignment(current_assignment, weights, constraints):
                break
            if current_assignment[i] >= 0:
                current_assignment[i] = -1

    current_value = objective_value(current_assignment, values)
    best_assignment = current_assignment.copy()
    best_value: np.int64 = np.int64(current_value)

    # (item_idx, target_k) -> expire_iter (tabu-until iteration)
    tabu_until: Dict[Tuple[int, int], int] = {}

    trajectory: List[np.int64] = [best_value]

    for iteration in range(1, max_iterations + 1):
        loads = compute_loads(current_assignment, weights, K)

        items = np.arange(N)
        rng.shuffle(items)
        if candidate_items is not None and candidate_items > 0:
            items = items[: min(candidate_items, N)]

        chosen_move: Optional[Tuple[int, int]] = None
        chosen_prev_k: int = -1
        chosen_new_value: np.int64 = np.int64(-1)

        for i in items:
            prev_k = int(current_assignment[i])
            targets = list(range(-1, K))
            rng.shuffle(targets)
            for target_k in targets:
                if target_k == prev_k:
                    continue
                if not _can_move_item(i, target_k, current_assignment, loads, weights, constraints):
                    continue

                if prev_k < 0 and target_k >= 0:
                    delta = np.int64(values[i])
                elif prev_k >= 0 and target_k < 0:
                    delta = -np.int64(values[i])
                else:
                    delta = np.int64(0)
                new_value = current_value + delta

                is_tabu = iteration < tabu_until.get((i, target_k), -1)
                if is_tabu and new_value <= best_value:
                    continue

                if new_value > chosen_new_value:
                    chosen_move = (i, target_k)
                    chosen_prev_k = prev_k
                    chosen_new_value = new_value

        if chosen_move is None:
            break

        i, target_k = chosen_move
        _apply_move(i, target_k, current_assignment, loads, weights)
        current_value = chosen_new_value

        tabu_until[(i, chosen_prev_k)] = iteration + tabu_tenure

        if current_value > best_value:
            best_value = current_value
            best_assignment = current_assignment.copy()

        trajectory.append(best_value)

    return best_assignment, best_value, trajectory


def iterated_local_search(
    values: np.ndarray,
    weights: np.ndarray,
    constraints: np.ndarray,
    rng: np.random.Generator,
    outer_iterations: int = 50,
    perturbation_strength: int = 3,
    max_hc_iterations: int = 10000,
) -> Tuple[np.ndarray, np.int64, List[np.int64]]:
    """Simple Iterated Local Search (ILS) using hill climbing as the local search.

    Procedure:
    1) Start from greedy solution; hill climb to local optimum.
    2) Perturb by dropping a small number of randomly selected assigned items.
    3) Re-apply hill climbing.
    4) Keep the best-so-far solution.
    """
    # Initial local search
    current_assignment, current_value, _ = hill_climbing(
        values, weights, constraints, rng, max_iterations=max_hc_iterations
    )
    best_assignment = current_assignment.copy()
    best_value = np.int64(current_value)
    trajectory: List[np.int64] = [best_value]

    for _ in range(outer_iterations):
        # Perturbation: randomly drop up to 'perturbation_strength' assigned items
        assigned_items = np.where(current_assignment >= 0)[0]
        if assigned_items.size > 0:
            num_drop = int(min(perturbation_strength, assigned_items.size))
            drop_indices = rng.choice(assigned_items, size=num_drop, replace=False)
            for i in drop_indices:
                current_assignment[i] = -1

        # Local search from perturbed state
        # Ensure feasibility is maintained by hill_climbing's internal checks
        current_assignment, current_value, _ = hill_climbing(
            values, weights, constraints, rng, max_iterations=max_hc_iterations
        )

        # Accept if better (basic acceptance criterion)
        if current_value > best_value:
            best_value = current_value
            best_assignment = current_assignment.copy()

        trajectory.append(best_value)

    return best_assignment, best_value, trajectory


def simulated_annealing(
    values: np.ndarray,
    weights: np.ndarray,
    constraints: np.ndarray,
    rng: np.random.Generator,
    max_iterations: int = 20000,
    start_temperature: float = 1.0,
    end_temperature: float = 0.01,
    candidate_items: int = 200,
) -> Tuple[np.ndarray, np.int64, List[np.int64]]:
    """Simulated Annealing with feasible single-item moves and exponential cooling.

    - State: feasible assignment.
    - Neighbor: move one item to another knapsack or drop; maintain feasibility.
    - Acceptance: always accept improving; accept worsening with probability exp(Δ/T).
    - Cooling: temperature decreases exponentially from start to end over iterations.
    """
    K, N, _D = get_problem_shape(values, weights, constraints)

    # Start from greedy feasible solution
    current_assignment = greedy_initialize(values, weights, constraints, rng)
    if not is_feasible_assignment(current_assignment, weights, constraints):
        densities = values / (1.0 + weights.sum(axis=1))
        for i in np.argsort(densities):
            if is_feasible_assignment(current_assignment, weights, constraints):
                break
            if current_assignment[i] >= 0:
                current_assignment[i] = -1

    current_value = objective_value(current_assignment, values)
    best_assignment = current_assignment.copy()
    best_value: np.int64 = np.int64(current_value)
    trajectory: List[np.int64] = [best_value]

    # Precompute cooling schedule parameters
    if max_iterations <= 1:
        alpha = 1.0
    else:
        alpha = (end_temperature / start_temperature) ** (1.0 / (max_iterations - 1))
    temperature = start_temperature

    for iteration in range(max_iterations):
        loads = compute_loads(current_assignment, weights, K)

        # Build a random candidate move
        items = np.arange(N)
        rng.shuffle(items)
        if candidate_items is not None and candidate_items > 0:
            items = items[: min(candidate_items, N)]

        move_found = False
        chosen_item = -1
        chosen_target = -1
        delta_value = np.int64(0)

        for i in items:
            prev_k = int(current_assignment[i])
            targets = list(range(-1, K))
            rng.shuffle(targets)
            for target_k in targets:
                if target_k == prev_k:
                    continue
                if not _can_move_item(i, target_k, current_assignment, loads, weights, constraints):
                    continue
                # Value change only when assigning or dropping
                if prev_k < 0 and target_k >= 0:
                    delta = np.int64(values[i])
                elif prev_k >= 0 and target_k < 0:
                    delta = -np.int64(values[i])
                else:
                    delta = np.int64(0)
                chosen_item = i
                chosen_target = target_k
                delta_value = delta
                move_found = True
                break
            if move_found:
                break

        if not move_found:
            # No feasible move found; stop
            break

        new_value = current_value + delta_value
        improve = new_value > current_value
        if improve:
            accept = True
        else:
            # Accept with SA probability
            # Note: delta_value may be negative; use float for prob
            prob = np.exp(float(delta_value) / max(temperature, 1e-12))
            accept = rng.random() < prob

        if accept:
            _apply_move(chosen_item, chosen_target, current_assignment, loads, weights)
            current_value = new_value
            if current_value > best_value:
                best_value = current_value
                best_assignment = current_assignment.copy()
        
        trajectory.append(best_value)

        # Cool down
        temperature *= alpha

    return best_assignment, best_value, trajectory


