import argparse
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
	from scipy.stats import f_oneway, friedmanchisquare, ttest_rel

	SCIPY_AVAILABLE = True
except Exception:
	SCIPY_AVAILABLE = False


# Reproducible optimization runs.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class Task:
	id: int
	cpu: float = 0.0
	depth: float = 0.0
	shots: float = 0.0
	qubits: float = 0.0
	deadline: float = 0.0
	max_price: float = 0.0
	coverage: list = field(default_factory=list)
	position: tuple = (0.0, 0.0)
	ram: float = 0.0
	bandwidth: float = 0.0


@dataclass
class UAV:
	id: int
	position: tuple = (0.0, 0.0)
	cpu: float = 0.0
	qubits: float = 0.0
	clops: float = 0.0
	ram: float = 0.0
	bandwidth: float = 0.0
	price_rate: float = 2e-4
	energy_coeff: float = 1.2e-6
	range: float = 0.0


@dataclass
class EdgeServer(UAV):
	pass


def initialize_tasks(df: pd.DataFrame) -> list[Task]:
	tasks = [Task(i) for i in range(len(df))]
	resources = df.iloc[:, 1:10].values
	positions = df.iloc[:, 10:12].values

	for i, t in enumerate(tasks):
		t.cpu = float(resources[i][0])
		t.ram = float(resources[i][1])
		t.bandwidth = float(resources[i][2])
		t.depth = float(resources[i][4])
		t.shots = float(resources[i][5])
		t.qubits = float(resources[i][6])
		t.deadline = float(resources[i][7])
		t.max_price = float(resources[i][8])
		t.position = (float(positions[i][0]), float(positions[i][1]))

	return tasks


def initialize_uavs(df: pd.DataFrame) -> list[UAV]:
	uavs = [UAV(i) for i in range(len(df))]
	resources = df.iloc[:, 1:7].values
	positions = df.iloc[:, 7:10].values

	for i, u in enumerate(uavs):
		u.cpu = float(resources[i][0])
		u.ram = float(resources[i][1])
		u.bandwidth = float(resources[i][2])
		u.qubits = float(resources[i][4])
		u.clops = float(resources[i][5])
		u.price_rate = 2e-4
		u.energy_coeff = 1.2e-6
		u.position = (float(positions[i][0]), float(positions[i][1]))
		u.range = float(positions[i][2])

	return uavs


def initialize_servers(df: pd.DataFrame) -> list[EdgeServer]:
	servers = [EdgeServer(i) for i in range(len(df))]
	resources = df.iloc[:, 1:7].values
	positions = df.iloc[:, 7:10].values

	for i, s in enumerate(servers):
		s.cpu = float(resources[i][0])
		s.ram = float(resources[i][1])
		s.bandwidth = float(resources[i][2])
		s.qubits = float(resources[i][4])
		s.clops = float(resources[i][5])
		s.price_rate = 3e-4
		s.energy_coeff = 1.8e-6
		s.position = (float(positions[i][0]), float(positions[i][1]))
		s.range = float(positions[i][2])

	return servers


def distance(point_a: tuple, point_b: tuple) -> float:
	return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def initialize_coverages(tasks: list[Task], processors: list[UAV]) -> None:
	for t in tasks:
		for p in processors:
			if distance(t.position, p.position) < p.range:
				t.coverage.append(p)


def create_lambda_matrix(processors: list[UAV], tasks: list[Task]) -> np.ndarray:
	lam = np.zeros((len(tasks), len(processors)), dtype=int)
	for i, t in enumerate(tasks):
		for j, p in enumerate(processors):
			if p in t.coverage and p.qubits >= t.qubits and p.ram >= t.ram and p.bandwidth >= t.bandwidth:
				lam[i][j] = 1
	return lam


def quantum_population_angles(pop_size: int, dim: int) -> np.ndarray:
	theta_pop = np.zeros((pop_size, dim), dtype=float)
	for i in range(pop_size):
		for j in range(dim):
			x = random.uniform(-1e6, 1e6)
			y = random.uniform(-1e6, 1e6)
			norm = math.sqrt((x * x) + (y * y))
			if norm == 0:
				alpha, beta = 1.0, 0.0
			else:
				alpha, beta = x / norm, y / norm
			theta_pop[i, j] = math.atan2(beta, alpha)
	return theta_pop


def observe(theta_vec: np.ndarray) -> list[int]:
	bits = []
	for theta in theta_vec:
		prob_zero = math.cos(theta) ** 2
		bits.append(0 if random.random() <= prob_zero else 1)
	return bits


def has_sufficient_resources(task: Task, processor: UAV) -> bool:
	return (
		processor in task.coverage
		and processor.qubits >= task.qubits
		and processor.ram >= task.ram
		and processor.bandwidth >= task.bandwidth
	)


def hashing(pi: int, task: Task, processor_index: dict[int, int]) -> int:
	candidates = list(task.coverage)
	if not candidates:
		return -1

	if 0 <= pi < len(candidates) and has_sufficient_resources(task, candidates[pi]):
		return processor_index[id(candidates[pi])]

	valid = False
	while (not valid) and candidates:
		h = pi % len(candidates)
		if has_sufficient_resources(task, candidates[h]):
			valid = True
			return processor_index[id(candidates[h])]
		del candidates[h]
		pi -= 1

	return -1


def decode(bits: list[int], bits_per_task: int, tasks: list[Task], processor_index: dict[int, int]) -> list[int]:
	decoded = []
	base = 2 ** (bits_per_task - 1)
	pi = 0
	task_id = 0

	for d in range(len(bits)):
		pi += int(bits[d]) * base
		base //= 2

		if (d + 1) % bits_per_task == 0:
			decoded.append(hashing(pi, tasks[task_id], processor_index))
			task_id += 1
			pi = 0
			base = 2 ** (bits_per_task - 1)

	return decoded


def objective_components(
	decoded_vector: list[int],
	tasks: list[Task],
	processors: list[UAV],
	no_of_uavs: int,
	no_of_servers: int,
) -> dict[str, float]:
	total_energy = 0.0
	allocated = 0
	active_server_ids: set[int] = set()
	load_count = np.zeros(len(processors), dtype=float)

	for t_idx, p_idx in enumerate(decoded_vector):
		if p_idx == -1:
			continue

		task = tasks[t_idx]
		proc = processors[p_idx]
		allocated += 1
		load_count[p_idx] += 1

		if p_idx >= no_of_uavs:
			active_server_ids.add(p_idx)

		work = max(1.0, float(task.cpu + (task.depth * task.shots)))
		service_rate = max(1.0, float(proc.clops))
		delay = work / service_rate
		energy = float(proc.energy_coeff) * (service_rate ** 2) * delay
		total_energy += energy

	uar = allocated / max(1, len(tasks))
	es_usage_ratio = len(active_server_ids) / max(1, no_of_servers)
	load_balance_std = float(np.std(load_count))
	energy_per_allocated = total_energy / max(1, allocated)

	return {
		"fitness_uar": float(uar),
		"fitness_es_usage_ratio": float(es_usage_ratio),
		"fitness_load_balance_std": float(load_balance_std),
		"fitness_energy_per_allocated": float(energy_per_allocated),
		"energy": float(total_energy),
		"allocated": float(allocated),
		"active_servers": float(len(active_server_ids)),
	}


def sa_cost_breakdown(
	decoded_vector: list[int],
	tasks: list[Task],
	processors: list[UAV],
	weights: dict[str, float],
	penalties: dict[str, float],
) -> dict[str, float]:
	total_delay = 0.0
	total_energy = 0.0
	penalty = 0.0

	for t_idx, p_idx in enumerate(decoded_vector):
		task = tasks[t_idx]
		if p_idx == -1:
			penalty += penalties["unallocated"]
			continue

		proc = processors[p_idx]
		work = max(1.0, float(task.cpu + (task.depth * task.shots)))
		service_rate = max(1.0, float(proc.clops))
		delay = work / service_rate
		energy = float(proc.energy_coeff) * (service_rate ** 2) * delay
		price = float(proc.price_rate) * work

		total_delay += delay
		total_energy += energy

		if delay > task.deadline:
			penalty += penalties["deadline"] * ((delay - task.deadline) / max(1.0, task.deadline))
		if price > task.max_price:
			penalty += penalties["budget"] * ((price - task.max_price) / max(1.0, task.max_price))

	weighted_delay = weights["delay"] * total_delay
	weighted_energy = weights["energy"] * total_energy
	total_cost = weighted_delay + weighted_energy + penalty

	return {
		"weighted_delay": float(weighted_delay),
		"weighted_energy": float(weighted_energy),
		"penalty": float(penalty),
		"total_delay": float(total_delay),
		"total_energy": float(total_energy),
		"total_cost": float(total_cost),
	}


def _task_security_level(task: Task) -> int:
	if task.qubits >= 20:
		return 2
	if task.qubits >= 10:
		return 1
	return 0


def identify_critical_tasks(tasks: list[Task], processors: list[UAV], feasible_map: list[list[int]]) -> list[bool]:
	local_energy = []
	avg_delay = []
	out_degree_proxy = []

	for t_idx, task in enumerate(tasks):
		feasible = feasible_map[t_idx]
		if not feasible:
			local_energy.append(0.0)
			avg_delay.append(float("inf"))
			out_degree_proxy.append(0.0)
			continue

		uav_first = [p for p in feasible if isinstance(processors[p], UAV) and not isinstance(processors[p], EdgeServer)]
		local_idx = uav_first[0] if uav_first else feasible[0]
		proc_local = processors[local_idx]
		work = max(1.0, float(task.cpu + (task.depth * task.shots)))
		delay_local = work / max(1.0, float(proc_local.clops))
		energy_local = float(proc_local.energy_coeff) * (max(1.0, float(proc_local.clops)) ** 2) * delay_local

		delays = [work / max(1.0, float(processors[p_idx].clops)) for p_idx in feasible]
		local_energy.append(float(energy_local))
		avg_delay.append(float(np.mean(delays)))
		out_degree_proxy.append(float(len(task.coverage)))

	total_energy = max(1e-9, float(sum(local_energy)))
	total_out = max(1e-9, float(sum(out_degree_proxy)))

	# Thresholds taken in the spirit of Eq. (19): critical if any ratio is high.
	k_e = 0.20
	k_o = 0.22
	k_d = 0.70

	critical = []
	for i, task in enumerate(tasks):
		energy_ratio = local_energy[i] / total_energy
		lat_ratio = avg_delay[i] / max(1e-9, float(task.deadline))
		out_ratio = out_degree_proxy[i] / total_out
		critical.append((energy_ratio > k_e) or (out_ratio > k_o) or (lat_ratio > k_d))

	return critical


def qde_ua(
	tasks: list[Task],
	processors: list[UAV],
	no_of_uavs: int,
	no_of_servers: int,
	bits_per_task: int,
	n_pop: int,
	g_max: int,
	f_scale: float,
	c_rate: float,
	weights: dict[str, float],
	verbose: bool = True,
) -> tuple[list[int], float, list[float], list[float], list[float]]:
	_ = bits_per_task
	_ = f_scale
	_ = c_rate
	penalties = {"deadline": 100.0, "budget": 100.0, "unallocated": 250.0}

	uav_ids = [i for i in range(no_of_uavs)]
	server_ids = [i for i in range(no_of_uavs, no_of_uavs + no_of_servers)]

	feasible_map: list[list[int]] = []
	for task in tasks:
		feasible = []
		for p_idx, proc in enumerate(processors):
			if has_sufficient_resources(task, proc):
				feasible.append(p_idx)
		feasible_map.append(feasible)

	critical_flags = identify_critical_tasks(tasks, processors, feasible_map)
	security_levels = [_task_security_level(t) for t in tasks]
	fixed_remote = [critical_flags[i] and (security_levels[i] != 0) for i in range(len(tasks))]

	def init_partition_solution() -> list[int]:
		sol = []
		for i, feasible in enumerate(feasible_map):
			if not feasible:
				sol.append(-1)
				continue

			if fixed_remote[i]:
				remote_feasible = [p for p in feasible if p in server_ids]
				sol.append(random.choice(remote_feasible if remote_feasible else feasible))
			else:
				local_feasible = [p for p in feasible if p in uav_ids]
				sol.append(random.choice(local_feasible if local_feasible else feasible))
		return sol

	def neighbor(solution: list[int]) -> list[int]:
		new_sol = solution.copy()
		for _ in range(20):
			t_idx = random.randrange(len(new_sol))
			feasible = feasible_map[t_idx]
			if not feasible:
				continue

			if fixed_remote[t_idx]:
				candidates = [p for p in feasible if p in server_ids]
				if not candidates:
					candidates = feasible
			else:
				candidates = feasible

			if len(candidates) == 1 and candidates[0] == new_sol[t_idx]:
				continue

			choices = [c for c in candidates if c != new_sol[t_idx]]
			new_sol[t_idx] = random.choice(choices if choices else candidates)
			break
		return new_sol

	current = init_partition_solution()
	current_cost = sa_cost_breakdown(current, tasks, processors, weights, penalties)["total_cost"]
	best_solution = current.copy()
	best_cost = current_cost

	history: list[float] = []
	exploration_history: list[float] = []
	exploitation_history: list[float] = []

	temp = 200.0
	cooling_rate = 0.95

	for g in range(1, g_max + 1):
		accepted = 0
		for _ in range(n_pop):
			cand = neighbor(current)
			cand_cost = sa_cost_breakdown(cand, tasks, processors, weights, penalties)["total_cost"]
			delta = cand_cost - current_cost

			if delta < 0 or random.random() < math.exp(-delta / max(1e-9, temp)):
				current = cand
				current_cost = cand_cost
				accepted += 1

			if cand_cost < best_cost:
				best_solution = cand.copy()
				best_cost = cand_cost

		history.append(float(best_cost))
		exploration_rate = accepted / max(1, n_pop)
		exploitation_rate = 1.0 - exploration_rate
		exploration_history.append(float(exploration_rate))
		exploitation_history.append(float(exploitation_rate))

		if verbose and (g % 5 == 0 or g == 1 or g == g_max):
			print(f"Iteration {g:>3}: best objective = {best_cost:.6f}")

		temp *= cooling_rate

	return best_solution, float(best_cost), history, exploration_history, exploitation_history


def processor_allocation_binary(decoded_vector: list[int], no_of_processors: int) -> str:
	used_processors = set(p_idx for p_idx in decoded_vector if p_idx != -1)
	return "".join("1" if p_idx in used_processors else "0" for p_idx in range(no_of_processors))


def build_task_result_frame(decoded_vector: list[int], tasks: list[Task], processors: list[UAV]) -> pd.DataFrame:
	records = []
	for t_idx, p_idx in enumerate(decoded_vector):
		task = tasks[t_idx]

		if p_idx == -1:
			records.append(
				{
					"task_id": t_idx + 1,
					"assigned_processor": -1,
					"processor_type": "NONE",
					"delay": np.nan,
					"price": np.nan,
					"energy": np.nan,
					"deadline": task.deadline,
					"max_price": task.max_price,
					"deadline_ok": False,
					"budget_ok": False,
				}
			)
			continue

		proc = processors[p_idx]
		work = max(1.0, float(task.cpu + (task.depth * task.shots)))
		service_rate = max(1.0, float(proc.clops))
		delay = work / service_rate
		energy = float(proc.energy_coeff) * (service_rate ** 2) * delay
		price = float(proc.price_rate) * work

		records.append(
			{
				"task_id": t_idx + 1,
				"assigned_processor": p_idx,
				"processor_type": "UAV" if isinstance(proc, UAV) and not isinstance(proc, EdgeServer) else "SERVER",
				"delay": delay,
				"price": price,
				"energy": energy,
				"deadline": task.deadline,
				"max_price": task.max_price,
				"deadline_ok": delay <= task.deadline,
				"budget_ok": price <= task.max_price,
			}
		)

	return pd.DataFrame(records)


def evaluate_decoded_solution(
	decoded_vector: list[int],
	tasks: list[Task],
	processors: list[UAV],
	no_of_uavs: int,
	no_of_servers: int,
) -> dict:
	comp = objective_components(decoded_vector, tasks, processors, no_of_uavs, no_of_servers)
	no_of_tasks = len(tasks)
	no_of_processors = len(processors)

	allocated_count = int(comp["allocated"])
	unallocated_count = no_of_tasks - allocated_count
	active_count = len(set(x for x in decoded_vector if x != -1))
	binary_alloc = processor_allocation_binary(decoded_vector, no_of_processors)
	avg_energy = comp["energy"] / max(1, allocated_count)

	results_df = build_task_result_frame(decoded_vector, tasks, processors)
	allocated_df = results_df[results_df["assigned_processor"] != -1].copy()

	total_budget_limit = float(results_df["max_price"].sum())
	total_deadline_limit = float(results_df["deadline"].sum())
	total_budget_used = float(allocated_df["price"].sum()) if not allocated_df.empty else 0.0
	total_delay_used = float(allocated_df["delay"].sum()) if not allocated_df.empty else 0.0

	if allocated_df.empty:
		total_budget_overrun = 0.0
		total_deadline_overrun = 0.0
	else:
		budget_diff = allocated_df["price"] - allocated_df["max_price"]
		deadline_diff = allocated_df["delay"] - allocated_df["deadline"]
		total_budget_overrun = float(np.clip(budget_diff, 0.0, None).sum())
		total_deadline_overrun = float(np.clip(deadline_diff, 0.0, None).sum())

	coverage_rate = (allocated_count / max(1, no_of_tasks)) * 100.0
	budget_violations = int((allocated_df["budget_ok"] == False).sum()) if not allocated_df.empty else 0
	deadline_violations = int((allocated_df["deadline_ok"] == False).sum()) if not allocated_df.empty else 0
	budget_violation_rate = (budget_violations / max(1, allocated_count)) * 100.0
	deadline_violation_rate = (deadline_violations / max(1, allocated_count)) * 100.0
	processor_utilization_rate = (active_count / max(1, no_of_processors)) * 100.0
	resource_utilization = active_count / max(1, no_of_processors)
	avg_delay = total_delay_used / max(1, allocated_count)
	avg_price = total_budget_used / max(1, allocated_count)

	total_delay = total_delay_used
	total_energy = float(comp["energy"])
	total_carbon = total_energy
	total_price = total_budget_used

	maximum_delay = total_deadline_limit
	maximum_energy = max(total_energy, 1e-9)
	maximum_carbon = max(total_carbon, 1e-9)
	maximum_price = max(total_budget_limit, 1e-9)
	maximum_resource_utilization = 1.0

	normalized_delay = float(np.clip(total_delay / max(1e-9, maximum_delay), 0.0, 1.0))
	normalized_energy = float(np.clip(total_energy / maximum_energy, 0.0, 1.0))
	normalized_carbon = float(np.clip(total_carbon / maximum_carbon, 0.0, 1.0))
	normalized_resource_utilization = float(np.clip(resource_utilization / maximum_resource_utilization, 0.0, 1.0))
	normalized_price = float(np.clip(total_price / maximum_price, 0.0, 1.0))
	random_weights = np.random.random(3)
	random_weights /= max(1e-9, float(np.sum(random_weights)))
	normalized_fitness = float(
		np.clip(
			(random_weights[0] * normalized_delay)
			+ (random_weights[1] * normalized_energy)
			+ (random_weights[2] * normalized_price),
			0.0,
			1.0,
		)
	)

	return {
		"comp": comp,
		"results_df": results_df,
		"allocated_count": allocated_count,
		"unallocated_count": unallocated_count,
		"active_count": active_count,
		"binary_alloc": binary_alloc,
		"avg_energy": avg_energy,
		"total_budget_limit": total_budget_limit,
		"total_deadline_limit": total_deadline_limit,
		"total_budget_used": total_budget_used,
		"total_delay_used": total_delay_used,
		"total_budget_overrun": total_budget_overrun,
		"total_deadline_overrun": total_deadline_overrun,
		"coverage_rate": coverage_rate,
		"budget_violations": budget_violations,
		"deadline_violations": deadline_violations,
		"budget_violation_rate": budget_violation_rate,
		"deadline_violation_rate": deadline_violation_rate,
		"processor_utilization_rate": processor_utilization_rate,
		"resource_utilization": resource_utilization,
		"avg_delay": avg_delay,
		"avg_price": avg_price,
		"total_delay": total_delay,
		"total_energy": total_energy,
		"total_carbon": total_carbon,
		"total_price": total_price,
		"maximum_delay": maximum_delay,
		"maximum_energy": maximum_energy,
		"maximum_carbon": maximum_carbon,
		"maximum_price": maximum_price,
		"maximum_resource_utilization": maximum_resource_utilization,
		"normalized_delay": normalized_delay,
		"normalized_energy": normalized_energy,
		"normalized_carbon": normalized_carbon,
		"normalized_resource_utilization": normalized_resource_utilization,
		"normalized_price": normalized_price,
		"normalized_fitness": normalized_fitness,
	}


def run_statistical_tests(sim_df: pd.DataFrame) -> dict[str, float]:
	metrics = [
		sim_df["normalized_delay"].values,
		sim_df["normalized_energy"].values,
		sim_df["normalized_carbon"].values,
		sim_df["normalized_resource_utilization"].values,
		sim_df["normalized_price"].values,
	]

	if SCIPY_AVAILABLE:
		anova = f_oneway(*metrics)
		friedman = friedmanchisquare(*metrics)
		exp_test = ttest_rel(sim_df["exploration_mean"].values, sim_df["exploitation_mean"].values)
		return {
			"anova_f": float(anova.statistic),
			"anova_p": float(anova.pvalue),
			"friedman_chi2": float(friedman.statistic),
			"friedman_p": float(friedman.pvalue),
			"exp_t": float(exp_test.statistic),
			"exp_p": float(exp_test.pvalue),
		}

	return {
		"anova_f": float("nan"),
		"anova_p": float("nan"),
		"friedman_chi2": float("nan"),
		"friedman_p": float("nan"),
		"exp_t": float("nan"),
		"exp_p": float("nan"),
	}


def main(user_csv: str = "user_500.csv", server_csv: str = "server_6.csv", uav_csv: str = "uav_4.csv") -> None:
	base = Path(__file__).resolve().parent
	user_path = base / user_csv
	server_path = base / server_csv
	uav_path = base / uav_csv

	task_data = pd.read_csv(user_path)
	server_data = pd.read_csv(server_path)
	uav_data = pd.read_csv(uav_path)

	tasks = initialize_tasks(task_data)
	uavs = initialize_uavs(uav_data)
	servers = initialize_servers(server_data)
	processors = uavs + servers

	initialize_coverages(tasks, processors)
	lambda_matrix = create_lambda_matrix(processors, tasks)

	no_of_tasks = len(tasks)
	no_of_processors = len(processors)

	max_coverage = max((len(t.coverage) for t in tasks), default=1)
	bits_per_task = max(1, math.ceil(math.log2(max_coverage)))

	n_pop = 30
	g_max = 50
	f_scale = 0.7
	c_rate = 0.72

	# SA objective from the paper style: weighted delay + weighted energy + penalties.
	weights = {"delay": 0.60, "energy": 0.40}

	print("=== DATASET SUMMARY ===")
	print(f"Users (tasks): {no_of_tasks}")
	print(f"Processors total: {no_of_processors} (UAV={len(uavs)}, SERVER={len(servers)})")
	print(f"Max coverage set size M_e: {max_coverage}")
	print(f"Bits per task: {bits_per_task}")
	print(f"Feasible links in lambda matrix: {int(np.sum(lambda_matrix))}")
	print()

	n_sim_runs = 20
	sim_records = []
	best_payload = None
	all_start_time = time.perf_counter()

	for run_id in range(n_sim_runs):
		run_seed = SEED + run_id
		random.seed(run_seed)
		np.random.seed(run_seed)
		run_start = time.perf_counter()

		run_solution, run_objective, _, run_exploration, run_exploitation = qde_ua(
			tasks,
			processors,
			len(uavs),
			len(servers),
			bits_per_task,
			n_pop,
			g_max,
			f_scale,
			c_rate,
			weights,
			verbose=False,
		)
		run_elapsed = time.perf_counter() - run_start

		run_metrics = evaluate_decoded_solution(run_solution, tasks, processors, len(uavs), len(servers))
		sim_records.append(
			{
				"run_id": run_id + 1,
				"seed": run_seed,
				"objective": float(run_objective),
				"normalized_delay": float(run_metrics["normalized_delay"]),
				"normalized_energy": float(run_metrics["normalized_energy"]),
				"normalized_carbon": float(run_metrics["normalized_carbon"]),
				"normalized_resource_utilization": float(run_metrics["normalized_resource_utilization"]),
				"normalized_price": float(run_metrics["normalized_price"]),
				"normalized_fitness": float(run_metrics["normalized_fitness"]),
				"exploration_mean": float(np.mean(run_exploration)),
				"exploitation_mean": float(np.mean(run_exploitation)),
				"computation_time": float(run_elapsed),
			}
		)

		if best_payload is None or run_objective < best_payload["best_objective"]:
			best_payload = {
				"best_solution": run_solution,
				"best_objective": float(run_objective),
				"exploration_history": run_exploration,
				"exploitation_history": run_exploitation,
				"metrics": run_metrics,
				"elapsed_time": run_elapsed,
				"run_id": run_id + 1,
				"seed": run_seed,
			}

	elapsed_time = time.perf_counter() - all_start_time
	sim_df = pd.DataFrame(sim_records)
	sim_output_path = base / "rp1_simulation_runs.csv"
	sim_df.to_csv(sim_output_path, index=False)
	stats = run_statistical_tests(sim_df)

	best_solution = best_payload["best_solution"]
	best_objective = best_payload["best_objective"]
	exploration_history = best_payload["exploration_history"]
	exploitation_history = best_payload["exploitation_history"]
	rm = best_payload["metrics"]

	comp = rm["comp"]
	allocated_count = rm["allocated_count"]
	unallocated_count = rm["unallocated_count"]
	active_count = rm["active_count"]
	binary_alloc = rm["binary_alloc"]
	avg_energy = rm["avg_energy"]

	sa_breakdown = sa_cost_breakdown(best_solution, tasks, processors, weights, {"deadline": 100.0, "budget": 100.0, "unallocated": 250.0})

	print("\n=== OPTIMIZATION RESULT ===")
	print(f"Fitness (objective): {best_objective:.6f}")
	print("Objective = w_delay*TotalDelay + w_energy*TotalEnergy + Penalty")
	print(f"Weighted delay term: {sa_breakdown['weighted_delay']:.6f}")
	print(f"Weighted energy term: {sa_breakdown['weighted_energy']:.6f}")
	print(f"Penalty term: {sa_breakdown['penalty']:.6f}")
	print(f"Best run ID / seed: {best_payload['run_id']} / {best_payload['seed']}")
	print(f"Allocated users: {allocated_count}/{no_of_tasks}")
	print(f"Unallocated users: {unallocated_count}")
	print(f"Active processors: {active_count}/{no_of_processors}")
	print(f"Active edge servers: {int(comp['active_servers'])}/{len(servers)}")
	print(f"Processor allocation (binary): {binary_alloc}")
	print(f"User allocation rate (UAR): {comp['fitness_uar'] * 100:.2f}%")
	print(f"Edge server usage ratio: {comp['fitness_es_usage_ratio']:.6f}")
	print(f"Load balance std-dev (task count): {comp['fitness_load_balance_std']:.6f}")
	print(f"Total energy: {comp['energy']:.6f}")
	print(f"Average energy per allocated user: {avg_energy:.6f}")

	print("\n=== PROCESSOR DETAILS ===")
	usage_count = {p_idx: 0 for p_idx in range(no_of_processors)}
	for p_idx in best_solution:
		if p_idx != -1:
			usage_count[p_idx] += 1

	for p_idx in range(no_of_processors):
		status = "Allocated" if usage_count[p_idx] > 0 else "Not Allocated"
		proc_type = "UAV" if p_idx < len(uavs) else "SERVER"
		print(f"Processor {p_idx:>2} ({proc_type}): {status}, tasks={usage_count[p_idx]}")

	results_df = rm["results_df"]
	total_budget_limit = rm["total_budget_limit"]
	total_deadline_limit = rm["total_deadline_limit"]
	total_budget_used = rm["total_budget_used"]
	total_delay_used = rm["total_delay_used"]
	total_budget_overrun = rm["total_budget_overrun"]
	total_deadline_overrun = rm["total_deadline_overrun"]
	coverage_rate = rm["coverage_rate"]
	budget_violations = rm["budget_violations"]
	deadline_violations = rm["deadline_violations"]
	budget_violation_rate = rm["budget_violation_rate"]
	deadline_violation_rate = rm["deadline_violation_rate"]
	processor_utilization_rate = rm["processor_utilization_rate"]
	resource_utilization = rm["resource_utilization"]
	avg_delay = rm["avg_delay"]
	avg_price = rm["avg_price"]
	total_delay = rm["total_delay"]
	total_energy = rm["total_energy"]
	total_carbon = rm["total_carbon"]
	total_price = rm["total_price"]
	un_normalized_fitness = best_objective
	maximum_delay = rm["maximum_delay"]
	maximum_energy = rm["maximum_energy"]
	maximum_carbon = rm["maximum_carbon"]
	maximum_price = rm["maximum_price"]
	maximum_resource_utilization = rm["maximum_resource_utilization"]
	normalized_delay = rm["normalized_delay"]
	normalized_energy = rm["normalized_energy"]
	normalized_carbon = rm["normalized_carbon"]
	normalized_resource_utilization = rm["normalized_resource_utilization"]
	normalized_price = rm["normalized_price"]
	best_fitness_value = rm["normalized_fitness"]

	print("\n=== COMPARISON METRICS ===")
	print(f"Total delay used (sum of Delay): {total_delay_used:.6f}")
	print(f"Average delay per allocated user: {avg_delay:.6f}")
	print(f"Total budget limit (sum of MaxPrice): {total_budget_limit:.6f}")
	print(f"Total budget used (sum of Price): {total_budget_used:.6f}")
	print(f"Average price per allocated user: {avg_price:.6f}")
	print(f"Total budget overrun amount: {total_budget_overrun:.6f}")
	print(f"Budget usage ratio (used/limit): {total_budget_used / max(1e-9, total_budget_limit):.6f}")
	print(f"Total deadline limit (sum of DeadLine): {total_deadline_limit:.6f}")
	print(f"Total deadline overrun amount: {total_deadline_overrun:.6f}")
	print(f"Delay usage ratio (used/limit): {total_delay_used / max(1e-9, total_deadline_limit):.6f}")
	print(f"Deadline violations: {deadline_violations}")
	print(f"Budget violations: {budget_violations}")
	print(f"Coverage rate: {coverage_rate:.2f}%")
	print(f"Processor utilization rate: {processor_utilization_rate:.2f}%")
	print(f"Budget violation rate (allocated users): {budget_violation_rate:.2f}%")
	print(f"Deadline violation rate (allocated users): {deadline_violation_rate:.2f}%")
	print(f"Normalized fitness (algorithm): {best_fitness_value:.6f}")

	print("\n=== STATISTICAL TESTS (ALL SIMULATION RUNS) ===")
	print(f"Simulation runs: {n_sim_runs}")
	print(f"ANOVA F-statistic: {stats['anova_f']:.6f}")
	print(f"ANOVA p-value: {stats['anova_p']:.6f}")
	print(f"Friedman chi-square: {stats['friedman_chi2']:.6f}")
	print(f"Friedman p-value: {stats['friedman_p']:.6f}")
	print(f"Exploration mean (runs): {sim_df['exploration_mean'].mean():.6f}")
	print(f"Exploitation mean (runs): {sim_df['exploitation_mean'].mean():.6f}")
	print(f"Exploration vs Exploitation t-stat: {stats['exp_t']:.6f}")
	print(f"Exploration vs Exploitation p-value: {stats['exp_p']:.6f}")

	output_path = base / "allocation_result.csv"
	results_df.to_csv(output_path, index=False)

	params_output_path = base / "output_research1.txt"
	with open(params_output_path, "w", encoding="utf-8") as file:
		file.write(f"NO OF ITERATIONS:\t{g_max}\n")
		file.write(f"NO OF USERS:\t{no_of_tasks}\n")
		file.write(f"NO OF PROCESSORS:\t{len(servers)} + {len(uavs)} = {no_of_processors}\n")
		file.write(f"RESOURCE UTILIZATION:\t{resource_utilization}\n")
		file.write(f"TOTAL DELAY:\t{total_delay}\n")
		file.write(f"TOTAL ENERGY:\t{total_energy}\n")
		file.write(f"CARBON FOOTPRINT:\t{total_carbon}\n")
		file.write(f"PRICE:\t{total_price}\n")
		file.write(f"FITNESS VALUE:\t{un_normalized_fitness}\n")
		file.write(f"NORMALIZED DELAY:\t{normalized_delay}\n")
		file.write(f"NORMALIZED ENERGY:\t{normalized_energy}\n")
		file.write(f"NORMALIZED CARBON:\t{normalized_carbon}\n")
		file.write(f"NORMALIZED RESOURCE UTILIZATION:\t{normalized_resource_utilization}\n")
		file.write(f"NORMALIZED PRICE:\t{normalized_price}\n")
		file.write(f"NORMALIZED FITNESS:\t{best_fitness_value}\n")
		file.write(f"MAXIMUM DELAY:\t{maximum_delay}\n")
		file.write(f"MAXIMUM ENERGY:\t{maximum_energy}\n")
		file.write(f"MAXIMUM CARBON:\t{maximum_carbon}\n")
		file.write(f"MAXIMUM PRICE:\t{maximum_price}\n")
		file.write(f"MAXIMUM RESOURCE UTILIZATION:\t{maximum_resource_utilization}\n")
		file.write(f"COMPUTATION TIME:\t{elapsed_time}\n")
		file.write(f"EXPLORATION:\t{exploration_history}\n")
		file.write(f"EXPLOITATION:\t{exploitation_history}\n")
		file.write(f"SIMULATION RUNS:\t{n_sim_runs}\n")
		file.write(f"ANOVA F-STAT:\t{stats['anova_f']}\n")
		file.write(f"ANOVA P-VALUE:\t{stats['anova_p']}\n")
		file.write(f"FRIEDMAN CHI2:\t{stats['friedman_chi2']}\n")
		file.write(f"FRIEDMAN P-VALUE:\t{stats['friedman_p']}\n")
		file.write(f"EXPLORATION MEAN (RUNS):\t{sim_df['exploration_mean'].mean()}\n")
		file.write(f"EXPLOITATION MEAN (RUNS):\t{sim_df['exploitation_mean'].mean()}\n")
		file.write(f"EXPLORATION-EXPLOITATION T-STAT:\t{stats['exp_t']}\n")
		file.write(f"EXPLORATION-EXPLOITATION P-VALUE:\t{stats['exp_p']}\n")

	print("\n=== TASK RESULT FILE ===")
	print(f"Saved task-wise allocation results to: {output_path.name}")
	print(f"Saved multi-run simulation table to: {sim_output_path.name}")
	print(f"Saved simulation summary to: {params_output_path.name}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run RP1 optimization with selectable dataset files.")
	parser.add_argument("--user-csv", default="user_500.csv", help="User/task dataset CSV filename in project root")
	parser.add_argument("--server-csv", default="server_6.csv", help="Server dataset CSV filename in project root")
	parser.add_argument("--uav-csv", default="uav_4.csv", help="UAV dataset CSV filename in project root")
	args = parser.parse_args()
	main(args.user_csv, args.server_csv, args.uav_csv)
