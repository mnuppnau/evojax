# Description: This file contains the implementation of the knowledge sources (KS) used in the knowledge-based optimization (KBO) framework.

from typing import Dict, List, Optional
import jax.numpy as jnp
import jax
import jax.tree_util
import jax.lax
import numpy as np
import copy

from jax import lax
from evojax.algo.cultural.population_space import Individual
from evojax.algo.cultural.helper_functions import (
    non_dominated_sort_lax,
    scale_arrays,
    calculate_slopes,
    calculate_slope,
    update_ks_weights,
    situational_score,
    historical_score,
    topographic_score,
    domain_score,
)


def initialize_domain_ks(param_size: int, num_solutions: int = 20):
    """Initialize Domain KS as a Pareto archive of non-dominated solutions.

    Stores the top non-dominated solutions across multiple objectives
    (adversarial, MI, entropy). This matches the structure expected by
    update_domain_ks which maintains a multi-objective Pareto front.
    """
    return (
        jnp.zeros((num_solutions, param_size)),  # best solutions (centers)
        jnp.zeros((num_solutions, param_size)),  # stdevs
        jnp.zeros((num_solutions, param_size)),  # scaled noises
        jnp.full((num_solutions,), 1000.0),      # adversarial fitness
        jnp.full((num_solutions,), 1000.0),      # MI fitness
        jnp.full((num_solutions,), 1000.0),      # Tchebycheff fitness
        jnp.full((num_solutions,), 1000.0),      # entropy
    )


def initialize_situational_ks(param_size: int):
    # Pre-allocate arrays with zeros for each individual property, given max_individuals
    # Assuming 'center' and 'stdev' are of size 'param_size'
    return (
        jnp.zeros((1, param_size)),  # parameter set, best individual 
        jnp.zeros((1, param_size)),  # stdev, best individual
        jnp.zeros((1, param_size)),  # scaled noise, best individual
        jnp.full((1,),1000),  # fitness values, adversarial
        jnp.full((1,),1000),  # fitness values, mutual information
        jnp.full((1,),1000),  # Tchebyschev fitness values
        jnp.full((1,),1000),  # entropy
    )


def initialize_history_ks(
    param_size: int, num_iterations: int = 40
):
    # Pre-allocate arrays with zeros for each individual property, given num_iterations
    # Assuming 'center' and 'stdev' are of size 'param_size'
    return (
        jnp.zeros((num_iterations, param_size)),  # best solutions
        jnp.zeros((num_iterations, param_size)),  # stdevs
        jnp.zeros((num_iterations, param_size)),  # scaled noises
        jnp.full((num_iterations,),1000),  # fitness values, adversarial
        jnp.full((num_iterations,),1000),  # fitness values, mutual information
        jnp.full((num_iterations,),1000),  # Tchebyschev fitness values
        jnp.full((num_iterations,),1000),
    )


def initialize_topographic_ks(
    features: int, key: jax.Array, num_codes: int = 11
):
        random_matrix = jax.random.normal(key, (features, num_codes))
        q_matrix, _ = jnp.linalg.qr(random_matrix)

        orthogonal_ks = q_matrix.T

        norms = jnp.linalg.norm(orthogonal_ks, axis=1, keepdims=True)
        normalized_ks = orthogonal_ks / norms
    
        velocity = jnp.zeros((num_codes, features))  # mnist centroids
        
        return ( 
            normalized_ks,  # orthogonal and normalized knowledge sources
            velocity,  # velocity for updating the knowledge sources
        )


def initialize_normative_ks(param_size: int, pop_size: int = 64):
    return (
        jnp.ones((pop_size,11,1)),  # pop_avg_spread
        jnp.ones((pop_size,11,11)),  # pop_min_safety

    )

@jax.jit
def update_domain_ks(
    belief_space, best_solution, stdev, best_scaled_noise, best_fitness_adv, best_fitness_mi, best_fitness_tchebycheff, disc_logit
):
    domain_ks = belief_space[1]

    best_solutions, stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, best_fitnesses_tchebycheff, entropies = domain_ks

    updated_best_solutions = jnp.concatenate([best_solutions, best_solution], axis=0)

    stdev = stdev.reshape(1, stdev.shape[0])
    updated_stdevs = jnp.concatenate([stdevs, stdev], axis=0)

    updated_best_scaled_noises = jnp.concatenate([best_scaled_noises, best_scaled_noise], axis=0)

    updated_best_fitness_adversarial = jnp.concatenate(
        [best_fitnesses_adversarial, best_fitness_adv.flatten()], axis=0
    )

    updated_best_fitness_mutual_info = jnp.concatenate(
        [best_fitnesses_mutual_info, best_fitness_mi.flatten()], axis=0
    )

    updated_best_fitness_tchebycheff = jnp.concatenate(
        [best_fitnesses_tchebycheff, best_fitness_tchebycheff.flatten()], axis=0
    )

    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)]) 

    #entropy = entropy.reshape(1, -1)

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitness_adversarial), abs(updated_best_fitness_mutual_info), abs(updated_entropy) ], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    # Select the top 20 non-dominated solutions
    num_selected = 20

    order = jnp.lexsort((-updated_best_fitness_adversarial, ranks))

    # select indices based on order
    selected_indices = order[:num_selected]
    #selected_indices = jnp.argsort(ranks)[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]

    selected_stdevs = updated_stdevs[selected_indices]

    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]    
    selected_best_fitness_adversarial = updated_best_fitness_adversarial[selected_indices]

    selected_best_fitness_mutual_info = updated_best_fitness_mutual_info[selected_indices]

    selected_best_fitness_tchebycheff = updated_best_fitness_tchebycheff[selected_indices]

    selected_entropy = updated_entropy[selected_indices]

    updated_domain_ks = (
        selected_best_solutions,
        selected_stdevs,
        selected_best_scaled_noises,
        selected_best_fitness_adversarial,
        selected_best_fitness_mutual_info,
        selected_best_fitness_tchebycheff,
        selected_entropy,
    )

    updated_belief_space_domain = (
        belief_space[:1] + (updated_domain_ks,) + belief_space[2:]
    )
    return updated_belief_space_domain

@jax.jit
def update_situational_ks(
    belief_space, solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, tchebyscheff_score, disc_logit
):
    situational_ks = belief_space[2]

    best_solution, best_stdev, best_scaled_noise, best_fitness_adversarial, best_fitness_mutual_info, best_fitness_tchebycheff, entropies = situational_ks

    updated_best_solution = jnp.concatenate([best_solution, solution], axis=0)
    
    stdev = stdev.reshape(1, stdev.shape[0])

    updated_best_stdev = jnp.concatenate([best_stdev, stdev], axis=0)
    updated_best_scaled_noise = jnp.concatenate([best_scaled_noise, scaled_noise], axis=0)
    
    updated_best_fitness_adversarial = jnp.concatenate(
        [best_fitness_adversarial, fitness_value_adv.flatten()], axis=0
    )
    updated_best_fitness_mutual_info = jnp.concatenate(
        [best_fitness_mutual_info, fitness_value_mi.flatten()], axis=0
    )
    updated_best_fitness_tchebycheff = jnp.concatenate(
        [best_fitness_tchebycheff, tchebyscheff_score.flatten()], axis=0
    )

    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack(
        [abs(updated_best_fitness_adversarial), abs(updated_best_fitness_mutual_info), abs(updated_best_fitness_tchebycheff), updated_entropy ]
    , axis=1)

    ranks = non_dominated_sort_lax(objectives)

    # Select the top solution

    num_selected = 1

    order = jnp.lexsort((-updated_best_fitness_adversarial, ranks))

    selected_indices = order[:num_selected]
    #selected_indices = jnp.argsort(ranks)[:num_selected]

    selected_best_solution = updated_best_solution[selected_indices]
    selected_best_stdev = updated_best_stdev[selected_indices]
    selected_best_scaled_noise = updated_best_scaled_noise[selected_indices]
    selected_best_fitness_adversarial = updated_best_fitness_adversarial[selected_indices]
    selected_best_fitness_mutual_info = updated_best_fitness_mutual_info[selected_indices]
    selected_best_fitness_tchebycheff = updated_best_fitness_tchebycheff[selected_indices]
    selected_disc_logits = updated_entropy[selected_indices]
    updated_situational_ks = (
        selected_best_solution,
        selected_best_stdev,
        selected_best_scaled_noise,
        selected_best_fitness_adversarial,
        selected_best_fitness_mutual_info,
        selected_best_fitness_tchebycheff,
        selected_disc_logits,
    )
    updated_belief_space_situational = (
        belief_space[:2] + (updated_situational_ks,) + belief_space[3:]
    )
    return updated_belief_space_situational

@jax.jit
def update_history_ks(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, tchebyscheff_score, disc_logit
):
    history_ks = belief_space[3]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, best_fitnesses_tchebycheff, entropies = history_ks

    updated_best_solutions = jnp.concatenate([best_solutions, best_solution], axis=0)
    
    stdev = stdev.reshape(1, stdev.shape[0])

    updated_best_stdevs = jnp.concatenate([best_stdevs, stdev], axis=0)
    updated_best_scaled_noises = jnp.concatenate([best_scaled_noises, scaled_noise], axis=0)
    
    updated_best_fitnesses_adversarial = jnp.concatenate(
        [best_fitnesses_adversarial, fitness_value_adv.flatten()], axis=0
    )
    updated_best_fitnesses_mutual_info = jnp.concatenate(
        [best_fitnesses_mutual_info, fitness_value_mi.flatten()], axis=0
    )
    updated_best_fitnesses_tchebycheff = jnp.concatenate(
        [best_fitnesses_tchebycheff, tchebyscheff_score.flatten()], axis=0
    )

    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    #entropy = entropy.reshape(-1, 1)
    
    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack(
        [abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy]
    , axis=1)
    
    ranks = non_dominated_sort_lax(objectives)

    # Select the top 40 non-dominated solutions
    num_selected = 40

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))
    selected_indices = order[:num_selected]
    #selected_indices = jnp.argsort(ranks)[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_best_fitnesses_tchebycheff = updated_best_fitnesses_tchebycheff[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_history_ks = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_best_fitnesses_tchebycheff,
        selected_entropy,
    )

    updated_belief_space_history = (
        belief_space[:3] + (updated_history_ks,) + belief_space[4:]
    )

    return updated_belief_space_history

@jax.jit
def update_topographic_ks(belief_space, avg_per_code, momentum=0.7):
    topographic_ks = belief_space[4]

    # 1. Retrieve History
    hist_avg_per_code = topographic_ks[0]     # Position at t-1
    hist_velocity = topographic_ks[1]         # Velocity at t-1 (Need to add this to state!)

    # 2. Update Position (EMA with configurable momentum)
    # Detect uninitialized state: velocity is all zeros after initialize_topographic_ks.
    # On first call (or after checkpoint restart), snap centroids to avg_per_code directly
    # so they reflect actual digit clusters, then use configured momentum afterward.
    is_initialized = jnp.any(hist_velocity != 0.0)
    effective_momentum = jnp.where(is_initialized, momentum, 0.0)

    # momentum=0.7 (default): 30% new data blended in each step (fast adaptation)
    # momentum=0.97+: centroids nearly frozen (prevents drift of locked codes)
    # momentum=0.0 (first call): snap to current avg_per_code
    updated_hist_avg_per_code = effective_momentum * hist_avg_per_code + (1.0 - effective_momentum) * avg_per_code

    # 3. Calculate Trend/Velocity (The "Prediction" Component)
    # How much did the centroid move this step?
    current_velocity = updated_hist_avg_per_code - hist_avg_per_code

    # Smooth the velocity to ignore jitter
    updated_velocity = 0.8 * hist_velocity + 0.2 * current_velocity
    
    # 4. Store Both
    # We store velocity so we can predict "Next Position = Current + Velocity"
    updated_topographic_ks = (
        updated_hist_avg_per_code,
        updated_velocity
    )

    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks,) + belief_space[5:]
    )

    return updated_belief_space_topographic

@jax.jit
def update_normative_ks(belief_space, fitness_scores, all_spreads, all_safety_ratios):
    # ... (Previous Elite Selection Logic) ...
    
    normative_ks = belief_space[5]

    k = 32
    elite_indices = jnp.argsort(fitness_scores)[-k:]
    
    # 1. Calculate Elite Metrics
    elite_ratios = all_safety_ratios[elite_indices] # (k, 10, 10)
    
    # IMPORTANT: We clamp the elite ratios BEFORE averaging.
    # If an elite has a ratio of 100.0, we treat it as 3.0.
    # This prevents one "infinite ratio" outlier from skewing the average.
    elite_ratios_clamped = jnp.minimum(elite_ratios, 3.0)
    
    min_ratios = jnp.min(elite_ratios_clamped, axis=(1,2))
    current_elite_min_safety = jnp.mean(min_ratios)
    
    # ... (Spread calculation remains the same) ...
    # 2. EXTRACT ELITE METRICS
    # Get spreads of elites
    elite_spreads = all_spreads[elite_indices] # (k, 10, 1)
    # Average spread across all codes for these elites
    current_elite_avg_spread = jnp.mean(elite_spreads)
    
    # Get safety ratios of elites
    elite_ratios = all_safety_ratios[elite_indices] # (k, 10, 10)
    # We care about the *worst* separation each elite had (the bottleneck)
    # But we average that bottleneck across the elite group
    # Logic: "What is the typical minimum safety margin for a top-tier agent?"
    min_ratios = jnp.min(elite_ratios, axis=(1,2)) # (k,)
    current_elite_min_safety = jnp.mean(min_ratios)
    
    # 2. Update Standards
    old_spread, old_safety = normative_ks
    
    new_spread = 0.95 * old_spread + 0.05 * current_elite_avg_spread
    new_safety = 0.95 * old_safety + 0.05 * current_elite_min_safety
    
    # 3. SAFETY CLAMPS (The Critical Fix)
    # Cap the maximum standard at 2.0.
    # Reason: A ratio of 2.0 means clusters are fully separated with a wide buffer.
    # Asking for more than 2.0 is mathematically unstable and unnecessary.
    new_safety = jnp.clip(new_safety, 1.1, 2.0)
    
    updated_normative_ks = ( 
        new_spread,
        new_safety
    )

    updated_belief_space_normative = (
        belief_space[:5] + (updated_normative_ks,)
    )

    return updated_belief_space_normative

@jax.jit
def get_center_guidance(belief_space, t, center):
    """Compute center guidance from all knowledge sources.

    Uses winner-take-all KS selection based on scoring functions.
    The KS scoring currently uses neutral slope defaults (0.0) since
    slope tracking is not yet stored in the normative KS. Once slope
    tracking is added, these should be replaced with real trend data.
    """
    domain_ks = belief_space[1]
    situational_ks = belief_space[2]
    history_ks = belief_space[3]
    topographic_ks = belief_space[4]

    # Neutral slope defaults — slope tracking not yet in normative KS.
    # These produce baseline scores so the winner-take-all still functions.
    adv_slope_short = jnp.float32(0.0)
    mi_slope_short = jnp.float32(0.0)
    adv_slope_med = jnp.float32(0.0)
    mi_slope_med = jnp.float32(0.0)
    entropy_long = jnp.float32(0.0)

    sit_score = situational_score(adv_slope_short, mi_slope_short)
    hist_score = historical_score(entropy_long, adv_slope_short)
    topo_score = topographic_score(entropy_long, adv_slope_med)
    dom_score = domain_score(adv_slope_med, mi_slope_med, entropy_long)

    ks_weights = jnp.array([dom_score, sit_score, hist_score, topo_score])

    # Winner-take-all: only the highest-scoring KS contributes
    max_index = jnp.argmax(ks_weights)
    ks_weights = jnp.zeros(4, dtype=jnp.int32).at[max_index].set(1)

    # Extract center guidance from each KS
    # Domain: best Pareto solution center
    domain_ks_center = domain_ks[0][0]
    # Situational: best overall solution center
    situational_ks_center = situational_ks[0]
    # Historical: solution with highest entropy (most diverse)
    history_max_entropy_idx = jnp.argmax(history_ks[6])
    history_ks_center = history_ks[0][history_max_entropy_idx]
    # Topographic: use the current PGPE center (topographic KS tracks
    # code centroids in feature space, not parameter space)
    topographic_ks_center = center

    domain_ks_center_weighted = domain_ks_center * ks_weights[0]
    situational_ks_center_weighted = situational_ks_center * ks_weights[1]
    history_ks_center_weighted = history_ks_center * ks_weights[2]
    topographic_ks_center_weighted = topographic_ks_center * ks_weights[3]

    return (
        domain_ks_center_weighted +
        situational_ks_center_weighted +
        history_ks_center_weighted +
        topographic_ks_center_weighted
    )

@jax.jit
def get_stdev_guidance(belief_space, t, stdev):
    """Compute stdev guidance from all knowledge sources.

    Uses winner-take-all KS selection based on scoring functions.
    See get_center_guidance for notes on slope defaults.
    """
    domain_ks = belief_space[1]
    situational_ks = belief_space[2]
    history_ks = belief_space[3]
    topographic_ks = belief_space[4]

    # Neutral slope defaults
    adv_slope_short = jnp.float32(0.0)
    mi_slope_short = jnp.float32(0.0)
    adv_slope_med = jnp.float32(0.0)
    mi_slope_med = jnp.float32(0.0)
    entropy_long = jnp.float32(0.0)

    sit_score = situational_score(adv_slope_short, mi_slope_short)
    hist_score = historical_score(entropy_long, adv_slope_short)
    topo_score = topographic_score(entropy_long, adv_slope_med)
    dom_score = domain_score(adv_slope_med, mi_slope_med, entropy_long)

    ks_weights = jnp.array([dom_score, sit_score, hist_score, topo_score])

    max_index = jnp.argmax(ks_weights)
    ks_weights = jnp.zeros(4, dtype=jnp.int32).at[max_index].set(1)

    # Extract stdev guidance from each KS
    domain_ks_stdev = domain_ks[1][0]
    situational_ks_stdev = situational_ks[1]
    history_ks_max_entropy_idx = jnp.argmax(history_ks[6])
    history_ks_stdev = history_ks[1][history_ks_max_entropy_idx]
    # Topographic: use current PGPE stdev as pass-through
    topographic_ks_stdev = stdev

    domain_ks_stdev_weighted = domain_ks_stdev * ks_weights[0]
    situational_ks_stdev_weighted = situational_ks_stdev * ks_weights[1]
    history_ks_stdev_weighted = history_ks_stdev * ks_weights[2]
    topographic_ks_stdev_weighted = topographic_ks_stdev * ks_weights[3]

    combined = (
        domain_ks_stdev_weighted +
        situational_ks_stdev_weighted +
        history_ks_stdev_weighted +
        topographic_ks_stdev_weighted
    )

    return combined, max_index
