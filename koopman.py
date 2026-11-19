import pandas as pd
import math
import numpy as np
import random
import time
from pathlib import Path

try:
    from scipy.stats import f_oneway, friedmanchisquare, ttest_rel

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =============================
# DATA
# =============================

tasks_data = [
    (50, 100),
    (40, 120),
    (60, 80),
    (30, 150),
    (45, 110)
]

processors = [200, 300, 600]
num_tasks = len(tasks_data)
Bits = 2
SEED = 42

# =============================
# FITNESS FUNCTION
# =============================

def compute_delay(task, proc):
    d, s = task
    return (d * s) / processors[proc]

def compute_energy(task, proc):
    delay = compute_delay(task, proc)
    return 1e-6 * (processors[proc]**2) * delay

def fitness(solution):
    total = 0
    for i, p in enumerate(solution):
        total += 0.5 * compute_energy(tasks_data[i], p)
        total += 0.3 * compute_delay(tasks_data[i], p)
    return total

# =============================
# SKIM + KOOPMAN
# =============================

def build_features(tasks):
    return np.array(tasks, dtype=float)

def encoder(X):
    return X / np.max(X)

def interaction(H):
    even = H[::2]
    odd = H[1::2]

    even_mean = np.mean(even, axis=0)

    for i in range(len(odd)):
        odd[i] = odd[i] * even_mean

    H_new = H.copy()
    H_new[1::2] = odd

    return H_new

def compute_koopman(X):
    X1 = X[:-1]
    X2 = X[1:]
    return X2.T @ np.linalg.pinv(X1).T

def apply_koopman(X, K):
    return (K @ X.T).T

def softmax(x):
    x = np.array(x)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def decision_layer(Z):

    input_dim = Z.shape[1]      # e.g., 2
    output_dim = 3              # number of processors

    # Xavier initialization (recommended)
    limit = np.sqrt(6 / (input_dim + output_dim))
    W = np.random.uniform(-limit, limit, (input_dim, output_dim))

    scores = Z @ W

    return scores

def skim_koopman_initialization(tasks_data):

    X = build_features(tasks_data)
    H = encoder(X)
    H = interaction(H)

    K = compute_koopman(H)
    Z = apply_koopman(H, K)

    scores = decision_layer(Z)

    prob_matrix = []

    for s in scores:
        probs = softmax(s)

        # safety normalization
        probs = probs / np.sum(probs)

        prob_matrix.append(probs)

    return prob_matrix

# =============================
# QUANTUM POPULATION
# =============================

def normalize_qubit(q):
    norm = math.sqrt(q[0]**2 + q[1]**2)
    return q / norm if norm > 0 else q

def quantum_population_skim(Psize, tasks, Bits, prob_matrix):

    D = len(tasks) * Bits
    Q = np.zeros((Psize, D, 2))

    for i in range(Psize):

        idx = 0

        for t in range(len(tasks)):

            # probs = prob_matrix[t]
            probs = prob_matrix[t % len(prob_matrix)]

            # safe sampling
            probs = np.array(probs) / np.sum(probs)
            chosen = np.random.choice(len(probs), p=probs)

            bits = [int(x) for x in format(chosen, f'0{Bits}b')]

            for b in bits:

                if b == 0:
                    alpha = np.sqrt(random.uniform(0.7, 1.0))
                    beta = np.sqrt(1 - alpha**2)
                else:
                    beta = np.sqrt(random.uniform(0.7, 1.0))
                    alpha = np.sqrt(1 - beta**2)

                Q[i, idx] = normalize_qubit(np.array([alpha, beta]))
                idx += 1

    return Q

# =============================
# OBSERVE
# =============================
prob_matrix = skim_koopman_initialization(tasks_data)


# =========================
# -------- CLASSES --------
# =========================
class Task:
    def __init__(self, index):
        self.id = index
        self.cpu = 0
        self.depth = 0
        self.shots = 0
        self.qubits = 0
        self.deadline = 0
        self.max_price = 0
        self.coverage = []
        self.position = None
        self.urgency = 0
        self.ram = 0
        self.bandwidth = 0

class SMD:
    def __init__(self, index):
        self.id = index
        self.qubits = 0
        self.clops = 0
        self.ram = 0
        self.bandwidth = 0

class UAV:
    def __init__(self, index):
        self.id = index
        self.position = None
        self.cpu = 0
        self.qubits = 0
        self.clops = 0
        self.ram = 0
        self.bandwidth = 0
        self.price_rate = 2e-4
        self.energy_coeff = 1e-6
        self.range = 0
        self.queue = []

    def enqueue_task(self, task):
        self.queue.append(task)

    def queue_length(self):
        return len(self.queue)

class EdgeServer(UAV):
    pass


# =========================
# ----- INITIALIZATION ----
# =========================
def initialize_tasks(df):
    task = [Task(i) for i in range(len(df))]
    resources = df.iloc[:, 1:10].values
    pos = df.iloc[:, 10:].values

    for i in range(len(task)):
        task[i].cpu = resources[i][0]
        task[i].ram = resources[i][1]
        task[i].bandwidth = resources[i][2]
        task[i].depth = resources[i][4]
        task[i].shots = resources[i][5]
        task[i].qubits = resources[i][6]
        task[i].deadline = resources[i][7]
        task[i].max_price = resources[i][8]
        task[i].position = (pos[i][0], pos[i][1])
    return task


def initialize_smd(df):
    smd = [SMD(i) for i in range(len(df))]
    resources = df.iloc[:, 12:].values
    for i in range(len(smd)):
        smd[i].ram = resources[i][1]
        smd[i].bandwidth = resources[i][2]
        smd[i].qubits = resources[i][3]
        smd[i].clops = resources[i][4]
    return smd


def initialize_uavs(df):
    uav = [UAV(i) for i in range(len(df))]
    res = df.iloc[:, 1:7].values
    pos = df.iloc[:, 7:].values

    for i in range(len(uav)):
        uav[i].cpu = res[i][0]
        uav[i].ram = res[i][1]
        uav[i].bandwidth = res[i][2]
        uav[i].qubits = res[i][4]
        uav[i].clops = res[i][5]
        uav[i].price_rate = 2e-4
        uav[i].energy_coeff = 1.2e-6
        uav[i].position = (pos[i][0], pos[i][1])
        uav[i].range = pos[i][2]
    return uav


def initialize_servers(df):
    srv = [EdgeServer(i) for i in range(len(df))]
    res = df.iloc[:, 1:7].values
    pos = df.iloc[:, 7:].values

    for i in range(len(srv)):
        srv[i].cpu = res[i][0]
        srv[i].ram = res[i][1]
        srv[i].bandwidth = res[i][2]
        srv[i].qubits = res[i][4]
        srv[i].clops = res[i][5]
        srv[i].price_rate = 3e-4
        srv[i].energy_coeff = 1.8e-6
        srv[i].position = (pos[i][0], pos[i][1])
        srv[i].range = pos[i][2]
    return srv


def initialize_processors(servers, uavs):
    return uavs + servers


# =========================
# -------- HELPERS --------
# =========================
def distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def initialize_coverages(tasks, processors):
    for t in tasks:
        for p in processors:
            if distance(*t.position, *p.position) < p.range:
                t.coverage.append(p)


def create_lambda_matrix(processors, tasks):
    lam = np.zeros((len(tasks), len(processors)))
    for i, t in enumerate(tasks):
        for j, p in enumerate(processors):
            if p in t.coverage:
                if p.qubits >= t.qubits and p.ram >= t.ram and p.bandwidth >= t.bandwidth:
                    lam[i][j] = 1
    return lam


# =========================
# ---- QUANTUM OPS --------
# =========================
def quantum_population_angles(pop, dim):
    theta_pop = np.zeros((pop, dim), dtype=float)
    for i in range(pop):
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


def observe(theta_vec):
    bv = []
    for theta in theta_vec:
        p0 = math.cos(theta) ** 2
        bv.append(0 if random.random() <= p0 else 1)
    return bv


def has_sufficient_resources(task, processor):
    return (
        processor in task.coverage
        and processor.qubits >= task.qubits
        and processor.ram >= task.ram
        and processor.bandwidth >= task.bandwidth
    )


def hashing(pi, task):
    candidates = list(task.coverage)

    if len(candidates) == 0:
        return -1

    if 0 <= pi < len(candidates) and has_sufficient_resources(task, candidates[pi]):
        return processor_index[id(candidates[pi])]

    flag = False
    while (not flag) and len(candidates) != 0:
        h = pi % len(candidates)
        if has_sufficient_resources(task, candidates[h]):
            flag = True
            return processor_index[id(candidates[h])]
        del candidates[h]
        pi = pi - 1

    return -1


def decode(bv):
    dv = []
    base = 2 ** (Bits - 1)
    pi = 0
    task_id = 0

    for d in range(len(bv)):
        pi += int(bv[d]) * base
        base = base // 2

        if (d + 1) % Bits == 0:
            dv.append(hashing(pi, tasks[task_id]))
            task_id += 1
            pi = 0
            base = 2 ** (Bits - 1)

    return dv


# =========================
# ----- FITNESS ----------
# =========================
def objective_components(decoded_vector):
    total_energy = 0.0
    total_delay = 0.0
    total_price = 0.0
    total_penalty = 0.0
    allocated = 0

    for t_idx, p_idx in enumerate(decoded_vector):
        task = tasks[t_idx]

        if p_idx == -1:
            total_penalty += PENALTY_UNALLOCATED
            continue

        proc = processors[p_idx]
        allocated += 1

        work = max(1.0, float(task.cpu + (task.depth * task.shots)))
        service_rate = max(1.0, float(proc.clops))
        delay = work / service_rate
        energy = float(proc.energy_coeff) * (service_rate ** 2) * delay
        price = float(proc.price_rate) * work

        total_delay += delay
        total_energy += energy
        total_price += price

        if delay > task.deadline:
            total_penalty += PENALTY_DEADLINE * ((delay - task.deadline) / max(1.0, task.deadline))

        if price > task.max_price:
            total_penalty += PENALTY_BUDGET * ((price - task.max_price) / max(1.0, task.max_price))

    return {
        "energy": total_energy,
        "delay": total_delay,
        "price": total_price,
        "penalty": total_penalty,
        "allocated": allocated,
    }


def fitness_function(decoded_vector):
    comp = objective_components(decoded_vector)
    return (
        (W_ENERGY * comp["energy"])
        + (W_DELAY * comp["delay"])
        + (W_PRICE * comp["price"])
        + comp["penalty"]
    )


# =========================
# -------- QDE-UA --------
# =========================
def QDE_UA(verbose=True):

    theta_pop = quantum_population_angles(N_pop, dim)
    decoded_population = []
    fit_population = []
    exploration_history = []
    exploitation_history = []

    for c in range(N_pop):
        bv = observe(theta_pop[c])
        dv = decode(bv)
        decoded_population.append(dv)
        fit_population.append(fitness_function(dv))

    for g in range(1, G_max + 1):
        accepted_trials = 0
        for c in range(N_pop):
            valid_ids = [idx for idx in range(N_pop) if idx != c]
            p, q, r = random.sample(valid_ids, 3)

            donor = theta_pop[p] + F_SCALE * (theta_pop[q] - theta_pop[r])
            cross_mask = np.random.rand(dim) <= C_RATE
            trial_theta = np.where(cross_mask, donor, theta_pop[c])

            trial_bv = observe(trial_theta)
            trial_dv = decode(trial_bv)
            trial_fit = fitness_function(trial_dv)

            if trial_fit <= fit_population[c]:
                theta_pop[c] = trial_theta
                decoded_population[c] = trial_dv
                fit_population[c] = trial_fit
                accepted_trials += 1

        best_idx = int(np.argmin(fit_population))
        exploration_rate = accepted_trials / max(1, N_pop)
        exploitation_rate = 1.0 - exploration_rate
        exploration_history.append(float(exploration_rate))
        exploitation_history.append(float(exploitation_rate))
        if verbose:
            print(f"Iteration {g}: {fit_population[best_idx]:.6f}")

    best_idx = int(np.argmin(fit_population))
    best_solution = decoded_population[best_idx]
    best_fitness_value = fit_population[best_idx]
    return best_solution, best_fitness_value, exploration_history, exploitation_history


def allocation_summary(decoded_vector):
    allocated = sum(1 for x in decoded_vector if x != -1)
    active = len(set(x for x in decoded_vector if x != -1))
    return allocated, active


def evaluate_solution(decoded_vector):
    comp = objective_components(decoded_vector)
    allocated_count, active_count = allocation_summary(decoded_vector)
    resource_utilization = active_count / max(1, no_of_processors)

    total_delay = float(comp["delay"])
    total_energy = float(comp["energy"])
    total_price = float(comp["price"])
    total_penalty = float(comp["penalty"])
    objective_value = float(fitness_function(decoded_vector))

    return {
        "allocated_count": int(allocated_count),
        "active_count": int(active_count),
        "resource_utilization": float(resource_utilization),
        "total_delay": total_delay,
        "total_energy": total_energy,
        "total_price": total_price,
        "total_penalty": total_penalty,
        "objective": objective_value,
    }


def run_statistical_tests(sim_df):
    metrics = [
        sim_df["normalized_delay"].values,
        sim_df["normalized_energy"].values,
        sim_df["normalized_price"].values,
        sim_df["normalized_penalty"].values,
        sim_df["normalized_objective"].values,
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


def processor_allocation_binary(decoded_vector):
    used_processors = set(p_idx for p_idx in decoded_vector if p_idx != -1)
    binary_string = ""
    for p_idx in range(no_of_processors):
        binary_string += "1" if p_idx in used_processors else "0"
    return binary_string


# =========================
# -------- MAIN ----------
# =========================
task_data = pd.read_csv('user_500.csv')
server_data = pd.read_csv('server_6.csv')
uav_data = pd.read_csv('uav_4.csv')

tasks = initialize_tasks(task_data)
smds = initialize_smd(task_data)
uavs = initialize_uavs(uav_data)
servers = initialize_servers(server_data)

processors = initialize_processors(servers, uavs)

initialize_coverages(tasks, processors)
lambda_matrix = create_lambda_matrix(processors, tasks)

no_of_tasks = len(tasks)
no_of_processors = len(processors)

M_e = 5
M_e = max(1, max((len(t.coverage) for t in tasks), default=1))
Bits = math.ceil(math.log2(M_e))
dim = no_of_tasks * Bits

N_pop = 30
G_max = 50
F_SCALE = 0.7
C_RATE = 0.72

W_ENERGY = 0.40
W_DELAY = 0.30
W_PRICE = 0.30

PENALTY_BUDGET = 100.0
PENALTY_DEADLINE = 100.0
PENALTY_UNALLOCATED = 250.0

processor_index = {id(p): i for i, p in enumerate(processors)}

best_solution = None
best_fitness_value = None

n_sim_runs = 20
sim_records = []
best_payload = None
overall_start = time.perf_counter()

for run_id in range(n_sim_runs):
    run_seed = SEED + run_id
    random.seed(run_seed)
    np.random.seed(run_seed)
    run_start = time.perf_counter()
    try:
        run_solution, run_fitness, run_exploration, run_exploitation = QDE_UA(verbose=False)
    except Exception as e:
        print(f"Error during QDE-UA run {run_id + 1}: {e}")
        continue

    run_elapsed = time.perf_counter() - run_start
    metrics = evaluate_solution(run_solution)
    sim_records.append(
        {
            "run_id": run_id + 1,
            "seed": run_seed,
            "objective": metrics["objective"],
            "total_delay": metrics["total_delay"],
            "total_energy": metrics["total_energy"],
            "total_price": metrics["total_price"],
            "total_penalty": metrics["total_penalty"],
            "resource_utilization": metrics["resource_utilization"],
            "exploration_mean": float(np.mean(run_exploration)),
            "exploitation_mean": float(np.mean(run_exploitation)),
            "computation_time": run_elapsed,
        }
    )

    if best_payload is None or run_fitness < best_payload["fitness"]:
        best_payload = {
            "run_id": run_id + 1,
            "seed": run_seed,
            "solution": run_solution,
            "fitness": run_fitness,
            "exploration": run_exploration,
            "exploitation": run_exploitation,
            "metrics": metrics,
        }

total_elapsed = time.perf_counter() - overall_start

sim_df = pd.DataFrame(sim_records)
if sim_df.empty:
    raise RuntimeError("No simulation runs completed successfully.")

sim_df["normalized_delay"] = sim_df["total_delay"] / max(1e-9, sim_df["total_delay"].max())
sim_df["normalized_energy"] = sim_df["total_energy"] / max(1e-9, sim_df["total_energy"].max())
sim_df["normalized_price"] = sim_df["total_price"] / max(1e-9, sim_df["total_price"].max())
sim_df["normalized_penalty"] = sim_df["total_penalty"] / max(1e-9, sim_df["total_penalty"].max())
sim_df["normalized_objective"] = sim_df["objective"] / max(1e-9, sim_df["objective"].max())

stats = run_statistical_tests(sim_df)

best_solution = best_payload["solution"]
best_fitness_value = best_payload["fitness"]

print("Best Objective:", best_fitness_value)
print("Best Fitness:", best_fitness_value)
normalized_best_fitness = 0.0 if best_fitness_value is None else 1.0 / (1.0 + float(best_fitness_value))
print("Normalized Fitness:", normalized_best_fitness)
if best_solution is not None:
    allocated_count, active_count = allocation_summary(best_solution)
    comp = objective_components(best_solution)
    binary_alloc = processor_allocation_binary(best_solution)
    print("Allocated Users:", allocated_count, "/", no_of_tasks)
    print("Active Processors:", active_count, "/", no_of_processors)
    print("Processor Allocation (Binary):", binary_alloc)
    print("Processor Details:")
    for p_idx in range(no_of_processors):
        status = "Allocated" if binary_alloc[p_idx] == "1" else "Not Allocated"
        print(f"  Processor {p_idx}: {binary_alloc[p_idx]} ({status})")
    print("Total Energy:", round(comp["energy"], 6))
    print("Total Delay:", round(comp["delay"], 6))
    print("Total Price:", round(comp["price"], 6))
    print("Total Penalty:", round(comp["penalty"], 6))

print("\n=== STATISTICAL TESTS (ALL SIMULATION RUNS) ===")
print("Simulation runs:", n_sim_runs)
print("Best run ID / seed:", best_payload["run_id"], "/", best_payload["seed"])
print("ANOVA F-statistic:", stats["anova_f"])
print("ANOVA p-value:", stats["anova_p"])
print("Friedman chi-square:", stats["friedman_chi2"])
print("Friedman p-value:", stats["friedman_p"])
print("Exploration mean (runs):", float(sim_df["exploration_mean"].mean()))
print("Exploitation mean (runs):", float(sim_df["exploitation_mean"].mean()))
print("Exploration vs Exploitation t-stat:", stats["exp_t"])
print("Exploration vs Exploitation p-value:", stats["exp_p"])

base = Path(__file__).resolve().parent
sim_csv = base / "koopman_simulation_runs.csv"
sim_df.to_csv(sim_csv, index=False)

summary_path = base / "output_koopman.txt"
with open(summary_path, "w", encoding="utf-8") as file:
    file.write(f"NO OF ITERATIONS:\t{G_max}\n")
    file.write(f"NO OF USERS:\t{no_of_tasks}\n")
    file.write(f"NO OF PROCESSORS:\t{no_of_processors}\n")
    file.write(f"FITNESS VALUE:\t{best_fitness_value}\n")
    file.write(f"NORMALIZED FITNESS:\t{normalized_best_fitness}\n")
    file.write(f"TOTAL DELAY:\t{best_payload['metrics']['total_delay']}\n")
    file.write(f"TOTAL ENERGY:\t{best_payload['metrics']['total_energy']}\n")
    file.write(f"PRICE:\t{best_payload['metrics']['total_price']}\n")
    file.write(f"TOTAL PENALTY:\t{best_payload['metrics']['total_penalty']}\n")
    file.write(f"RESOURCE UTILIZATION:\t{best_payload['metrics']['resource_utilization']}\n")
    file.write(f"COMPUTATION TIME:\t{total_elapsed}\n")
    file.write(f"EXPLORATION:\t{best_payload['exploration']}\n")
    file.write(f"EXPLOITATION:\t{best_payload['exploitation']}\n")
    file.write(f"SIMULATION RUNS:\t{n_sim_runs}\n")
    file.write(f"ANOVA F-STAT:\t{stats['anova_f']}\n")
    file.write(f"ANOVA P-VALUE:\t{stats['anova_p']}\n")
    file.write(f"FRIEDMAN CHI2:\t{stats['friedman_chi2']}\n")
    file.write(f"FRIEDMAN P-VALUE:\t{stats['friedman_p']}\n")
    file.write(f"EXPLORATION MEAN (RUNS):\t{float(sim_df['exploration_mean'].mean())}\n")
    file.write(f"EXPLOITATION MEAN (RUNS):\t{float(sim_df['exploitation_mean'].mean())}\n")
    file.write(f"EXPLORATION-EXPLOITATION T-STAT:\t{stats['exp_t']}\n")
    file.write(f"EXPLORATION-EXPLOITATION P-VALUE:\t{stats['exp_p']}\n")

print("\nSaved multi-run simulation table to:", sim_csv.name)
print("Saved simulation summary to:", summary_path.name)