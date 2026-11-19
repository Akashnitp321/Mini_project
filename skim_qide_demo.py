#!/usr/bin/env python3
"""
SKIM-QIDE Manual Implementation on Small Dataset
Demonstrates all algorithm components step-by-step with detailed output
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# DATASET SETUP
# =============================================================================

print("="*80)
print("SKIM-QIDE: Small Dataset Demonstration")
print("="*80)

# QEN Resources: 3 nodes
qen_resources = [
    {
        'id': 0,
        'n_q': 20,           # Qubits
        'f_clk': 5,          # Clock (MHz)
        'b_net': 200,        # Bandwidth (Mbps)
        's_mem': 4,          # Memory (GB)
        'c_unit': 0.05,      # Cost ($/qubit-cycle)
        'e_cpu': 5e-8        # Energy (J/qubit-cycle)
    },
    {
        'id': 1,
        'n_q': 15,
        'f_clk': 3,
        'b_net': 150,
        's_mem': 3,
        'c_unit': 0.08,
        'e_cpu': 8e-8
    },
    {
        'id': 2,
        'n_q': 10,
        'f_clk': 8,
        'b_net': 300,
        's_mem': 2,
        'c_unit': 0.02,
        'e_cpu': 2e-8
    }
]

# Task Parameters: 6 tasks
task_params = [
    {
        'id': 0,
        'w_q': 5,            # Qubits
        'w_dep': 50,         # Depth (layers)
        'w_mem': 1.0,        # Memory (GB)
        'd_dl': 100,         # Deadline (ms)
        'b_max': 3.0,        # Budget (\$)
        'p_pen': 0.5         # Penalty ($/ms violation)
    },
    {
        'id': 1,
        'w_q': 8,
        'w_dep': 75,
        'w_mem': 1.5,
        'd_dl': 120,
        'b_max': 4.0,
        'p_pen': 0.6
    },
    {
        'id': 2,
        'w_q': 3,
        'w_dep': 30,
        'w_mem': 0.5,
        'd_dl': 80,
        'b_max': 2.0,
        'p_pen': 0.4
    },
    {
        'id': 3,
        'w_q': 6,
        'w_dep': 60,
        'w_mem': 1.0,
        'd_dl': 110,
        'b_max': 3.5,
        'p_pen': 0.55
    },
    {
        'id': 4,
        'w_q': 4,
        'w_dep': 40,
        'w_mem': 0.8,
        'd_dl': 90,
        'b_max': 2.5,
        'p_pen': 0.45
    },
    {
        'id': 5,
        'w_q': 7,
        'w_dep': 70,
        'w_mem': 1.2,
        'd_dl': 115,
        'b_max': 3.5,
        'p_pen': 0.5
    }
]

# Coverage sets: which QENs can serve each task
coverage_sets = [
    [0, 1, 2],  # Task 0 can be served by QENs 0, 1, 2
    [0, 1],     # Task 1 can be served by QENs 0, 1
    [0, 1, 2],  # Task 2 can be served by QENs 0, 1, 2
    [1, 2],     # Task 3 can be served by QENs 1, 2
    [0, 2],     # Task 4 can be served by QENs 0, 2
    [0, 1]      # Task 5 can be served by QENs 0, 1
]

print("\nDataset Configuration:")
print(f"  - Number of QENs: {len(qen_resources)}")
print(f"  - Number of Tasks: {len(task_params)}")
print("\nQEN Details:")
for qen in qen_resources:
    print(f"  QEN {qen['id']}: {qen['n_q']} qubits, {qen['f_clk']} MHz, \${qen['c_unit']}/q-cyc")
print("\nTask Details:")
for task in task_params:
    print(f"  Task {task['id']}: {task['w_q']} qubits, depth {task['w_dep']}, deadline {task['d_dl']} ms")

# =============================================================================
# ALGORITHM COMPONENTS
# =============================================================================

def initialize_quantum_population(pop_size: int, dim: int, v_max: float = 1.0) -> np.ndarray:
    """Initialize quantum population with uniformly distributed angles"""
    population = np.zeros((pop_size, dim))
    
    for c in range(pop_size):
        for d in range(dim):
            x = np.random.uniform(-v_max, v_max)
            y = np.random.uniform(-v_max, v_max)
            
            norm = np.sqrt(x**2 + y**2)
            if norm > 0:
                alpha = x / norm
                beta = y / norm
            else:
                alpha, beta = 1.0, 0.0
            
            theta = np.arctan2(beta, alpha)
            population[c, d] = theta
    
    return population

def observe_quantum_vector(theta_vector: np.ndarray) -> np.ndarray:
    """Collapse quantum superposition to binary vector"""
    dim = len(theta_vector)
    binary_vector = np.zeros(dim, dtype=int)
    
    for d in range(dim):
        theta = theta_vector[d]
        prob_zero = np.cos(theta)**2
        
        if np.random.random() <= prob_zero:
            binary_vector[d] = 0
        else:
            binary_vector[d] = 1
    
    return binary_vector

def decode_with_hashing(binary_vector: np.ndarray, n_tasks: int, bits_per_task: int,
                       coverage_sets: List[List[int]], qen_resources: List[Dict],
                       task_params: List[Dict]) -> np.ndarray:
    """Decode binary vector to feasible task-QEN assignment using hashing"""
    assignment = np.full(n_tasks, -1, dtype=int)
    qen_loads = np.zeros(len(qen_resources))
    mem_loads = np.zeros(len(qen_resources))
    
    for j in range(n_tasks):
        start_bit = j * bits_per_task
        end_bit = start_bit + bits_per_task
        bits = binary_vector[start_bit:end_bit]
        
        # Decode to decimal
        if any(bits):
            index = int(''.join(map(str, bits)), 2)
        else:
            index = 0
        
        cov = coverage_sets[j]
        found = False
        attempts = 0
        max_attempts = len(cov)
        
        while not found and attempts < max_attempts:
            hash_idx = (index + attempts) % len(cov)
            qen_id = cov[hash_idx]
            
            w_q = task_params[j]['w_q']
            w_mem = task_params[j]['w_mem']
            qen_qubit_cap = qen_resources[qen_id]['n_q']
            qen_mem_cap = qen_resources[qen_id]['s_mem']
            
            if (qen_loads[qen_id] + w_q <= qen_qubit_cap and
                mem_loads[qen_id] + w_mem <= qen_mem_cap):
                
                assignment[j] = qen_id
                qen_loads[qen_id] += w_q
                mem_loads[qen_id] += w_mem
                found = True
            
            attempts += 1
    
    return assignment

def compute_fitness(assignment: np.ndarray, qen_resources: List[Dict], 
                   task_params: List[Dict], n_qens: int, n_tasks: int,
                   weights: Tuple) -> Tuple[float, Dict]:
    """Compute six-component fitness function"""
    components = {}
    
    # 1. Task Allocation Rate (TAR)
    n_assigned = np.sum(assignment >= 0)
    f1 = n_assigned / n_tasks
    components['tar'] = f1
    
    qen_loads = np.zeros(n_qens)
    total_delay = 0
    total_cost = 0
    total_energy = 0
    budget_violations = 0
    qen_active = np.zeros(n_qens, dtype=bool)
    
    # Compute per-task metrics
    for j in range(n_tasks):
        if assignment[j] >= 0:
            i = int(assignment[j])
            qen_active[i] = True
            
            w_dep = task_params[j]['w_dep']
            w_q = task_params[j]['w_q']
            b_net = qen_resources[i]['b_net']
            f_clk = qen_resources[i]['f_clk']
            
            transmission_delay = (w_dep * w_q) / b_net
            execution_delay = w_dep / f_clk
            queuing_delay = qen_loads[i] / 1.0
            
            delta_ij = transmission_delay + queuing_delay + execution_delay
            total_delay += delta_ij
            
            # Cost
            c_unit = qen_resources[i]['c_unit']
            kappa_ij = w_q * w_dep * c_unit
            total_cost += kappa_ij
            
            # SLA Penalty
            d_dl = task_params[j]['d_dl']
            p_pen = task_params[j]['p_pen']
            pi_ij = p_pen * max(0, delta_ij - d_dl)
            total_cost += pi_ij
            
            # Energy
            e_cpu = qen_resources[i]['e_cpu']
            xi_ij = w_q * w_dep * e_cpu
            total_energy += xi_ij
            
            # Budget check
            b_max = task_params[j]['b_max']
            if kappa_ij > b_max:
                budget_violations += 1
            
            qen_loads[i] += 1
    
    # 2. QEN Utilization
    f2 = np.sum(qen_active) / n_qens if n_qens > 0 else 0
    components['utilization'] = f2
    
    # 3. Total Delay
    f3 = total_delay if total_delay > 0 else 1e-6
    components['delay'] = f3
    
    # 4. Total Cost
    f4 = total_cost if total_cost > 0 else 1e-6
    components['cost'] = f4
    
    # 5. Load Balance
    mu_avg = n_assigned / n_qens if n_assigned > 0 else 1
    load_imbalance = np.sqrt(np.mean((mu_avg - qen_loads)**2))
    f5 = load_imbalance
    components['load_balance'] = f5
    
    # 6. Budget Violations
    f_pen = budget_violations
    components['penalty'] = f_pen
    
    # Composite fitness
    w1, w2, w3, w4, w5, w6 = weights
    
    eps = 1e-6
    fitness = (w1 * f1 + 
              (w2 / (f2 + eps) if f2 > 0 else 0) +
              (w3 / (f3 + eps) if f3 > 0 else 0) +
              (w4 / (f4 + eps) if f4 > 0 else 0) +
              (w5 / (f5 + eps) if f5 > 0 else 1) -
              w6 * f_pen)
    
    return fitness, components

def quantum_crossover(donor: np.ndarray, target: np.ndarray, cr: float) -> np.ndarray:
    """Quantum crossover operation"""
    trial = np.copy(target)
    for d in range(len(target)):
        if np.random.random() < cr:
            trial[d] = donor[d]
    return trial

def dmd_fitness_prediction(fitness_history: List[np.ndarray], window_size: int, 
                          rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply DMD to predict next-generation fitness landscape"""
    if len(fitness_history) < window_size + 1:
        return None, None, None
    
    history = np.array(fitness_history[-(window_size+1):])
    X = history[:-1, :].T
    X_next = history[1:, :].T
    
    try:
        A = X_next @ np.linalg.pinv(X)
        eigenvalues, modes = np.linalg.eig(A)
        
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        modes = modes[:, idx]
        
        eigenvalues = eigenvalues[:rank]
        modes = modes[:, :rank]
        
        x0 = history[-1, :]
        b = np.linalg.lstsq(modes, x0, rcond=None)[0]
        
        next_fitness = np.real(modes @ (b * eigenvalues))
        
        return next_fitness, eigenvalues, modes
    except:
        return None, None, None

def skim_mutation(population: np.ndarray, fitness_values: np.ndarray, 
                 fitness_history: List[np.ndarray],
                 f_base: float, tau: float, window_size: int, rank: int,
                 pop_size: int, dim: int) -> Tuple[np.ndarray, float]:
    """Apply SKIM mutation using Koopman-guided pivot"""
    pred_fitness, eigenvalues, modes = dmd_fitness_prediction(fitness_history, window_size, rank)
    
    if pred_fitness is None:
        c_star = np.argmax(fitness_values)
        f_k = f_base
    else:
        c_star = np.argmax(pred_fitness)
        lambda_1 = np.abs(eigenvalues[0])
        f_k = f_base * np.exp(-(lambda_1 - 1) / tau)
    
    donor_vectors = np.zeros_like(population)
    
    for c in range(pop_size):
        if c == c_star:
            donor_vectors[c] = population[c]
        else:
            available = [i for i in range(pop_size) if i != c and i != c_star]
            if len(available) >= 2:
                indices = np.random.choice(available, size=2, replace=False)
                q, r = indices
                donor_vectors[c] = population[c_star] + f_k * (population[q] - population[r])
            else:
                donor_vectors[c] = population[c]
    
    return donor_vectors, f_k

# =============================================================================
# MAIN SKIM-QIDE ALGORITHM
# =============================================================================

def skim_qide_main(generations: int = 50):
    """Run SKIM-QIDE on small dataset"""
    
    # Parameters000
    pop_size = 4
    n_qens = len(qen_resources)
    n_tasks = len(task_params)
    m_e = max(len(cov) for cov in coverage_sets)
    bits_per_task = int(np.ceil(np.log2(m_e)))
    dim = n_tasks * bits_per_task
    
    # Algorithm parameters
    f_base = 0.1
    cr = 0.72
    tau = 0.5
    window_size = 5
    rank = 2
    weights = (0.35, 0.10, 0.20, 0.20, 0.10, 0.05)
    
    print("\n" + "="*80)
    print("ALGORITHM INITIALIZATION")
    print("="*80)
    print(f"Population size: {pop_size}")
    print(f"Dimensions: {dim} (each task needs {bits_per_task} bits)")
    print(f"Generations: {generations}")
    print(f"Window size for DMD: {window_size}")
    
    # Initialize
    population = initialize_quantum_population(pop_size, dim)
    fitness_history = []
    convergence_history = []
    
    # Evaluate initial population
    print("\n" + "="*80)
    print("GENERATION 0: INITIAL EVALUATION")
    print("="*80)
    
    fitness_values = np.zeros(pop_size)
    assignments = []
    
    for c in range(pop_size):
        binary_vec = observe_quantum_vector(population[c])
        assignment = decode_with_hashing(binary_vec, n_tasks, bits_per_task,
                                        coverage_sets, qen_resources, task_params)
        fitness, components = compute_fitness(assignment, qen_resources, task_params,
                                             n_qens, n_tasks, weights)
        
        fitness_values[c] = fitness
        assignments.append(assignment)
        
        print(f"\nIndividual {c}:")
        print(f"  Assignment: {assignment}")
        print(f"  TAR: {components['tar']:.4f} | Delay: {components['delay']:.2f} ms | Cost: \${components['cost']:.2f}")
        print(f"  Fitness: {fitness:.6f}")
    
    fitness_history.append(fitness_values.copy())
    best_idx = np.argmax(fitness_values)
    best_fitness = fitness_values[best_idx]
    best_assignment = assignments[best_idx]
    convergence_history.append(best_fitness)
    
    print(f"\n{'='*80}")
    print(f"Generation 0 Best: Individual {best_idx} with fitness {best_fitness:.6f}")
    print(f"{'='*80}")
    
    # Main loop
    for g in range(1, generations + 1):
        print(f"\n{'='*80}")
        print(f"GENERATION {g}")
        print(f"{'='*80}")
        
        # SKIM Mutation
        donor_pop, f_k = skim_mutation(population, fitness_values, fitness_history,
                                      f_base, tau, window_size, rank, pop_size, dim)
        
        if g % 10 == 0 or g <= 3:
            print(f"Scaling factor F_K: {f_k:.6f}")
        
        # Crossover and Selection
        new_assignments = []
        
        for c in range(pop_size):
            # Crossover
            trial_qav = quantum_crossover(donor_pop[c], population[c], cr)
            
            # Observation and Decoding
            binary_vec = observe_quantum_vector(trial_qav)
            trial_assignment = decode_with_hashing(binary_vec, n_tasks, bits_per_task,
                                                  coverage_sets, qen_resources, task_params)
            trial_fitness, trial_comp = compute_fitness(trial_assignment, qen_resources,
                                                        task_params, n_qens, n_tasks, weights)
            
            # Selection
            if trial_fitness >= fitness_values[c]:
                population[c] = trial_qav
                fitness_values[c] = trial_fitness
                assignments[c] = trial_assignment
                action = "ACCEPT"
            else:
                action = "REJECT"
            
            if g % 10 == 0 or g <= 3:
                print(f"  Ind {c}: fitness {trial_fitness:.6f} ({action})")
        
        fitness_history.append(fitness_values.copy())
        
        # Update best
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_assignment = assignments[current_best_idx]
        
        convergence_history.append(best_fitness)
        
        if g % 10 == 0:
            print(f"\nBest fitness so far: {best_fitness:.6f}")
            _, best_comp = compute_fitness(best_assignment, qen_resources, task_params,
                                          n_qens, n_tasks, weights)
            print(f"  TAR: {best_comp['tar']:.4f}")
            print(f"  Delay: {best_comp['delay']:.2f} ms")
            print(f"  Cost: \${best_comp['cost']:.2f}")
    
    return best_assignment, best_fitness, convergence_history, fitness_history

# =============================================================================
# EXECUTION
# =============================================================================

print("\n" + "="*80)
print("RUNNING SKIM-QIDE WITH 50 GENERATIONS")
print("="*80)

best_assignment, best_fitness, convergence_history, fitness_history = skim_qide_main(generations=50)

# Final Results
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

_, final_components = compute_fitness(best_assignment, qen_resources, task_params,
                                      len(qen_resources), len(task_params),
                                      (0.35, 0.10, 0.20, 0.20, 0.10, 0.05))

print(f"\nBest Assignment Found:")
for qen_id in range(len(qen_resources)):
    tasks = [j for j in range(len(task_params)) if best_assignment[j] == qen_id]
    print(f"  QEN {qen_id}: Tasks {tasks if tasks else 'None'}")

print(f"\nPerformance Metrics:")
print(f"  Task Allocation Rate (TAR): {final_components['tar']:.4f} ({int(final_components['tar']*len(task_params))}/{len(task_params)} tasks)")
print(f"  Total Execution Delay: {final_components['delay']:.2f} ms")
print(f"  Total Cost + Penalty: \${final_components['cost']:.2f}")
print(f"  Load Balance (Imbalance): {final_components['load_balance']:.4f}")
print(f"  Budget Violations: {int(final_components['penalty'])}")
print(f"  Composite Fitness: {best_fitness:.6f}")

# Convergence plot
print(f"\n{'='*80}")
print("Generating convergence plot...")
print(f"{'='*80}")

try:
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history, 'b-', linewidth=2, label='Best Fitness')
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    plt.title('SKIM-QIDE Convergence on Small Dataset', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150)
    print("✓ Convergence plot saved as 'convergence.png'")
except Exception as e:
    print(f"✗ Could not generate plot: {e}")

print("\n" + "="*80)
print("Demonstration Complete!")
print("="*80)
