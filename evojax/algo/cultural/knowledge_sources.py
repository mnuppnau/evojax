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


def initialize_domain_ks(num_clusters: int, num_pixels: int, num_images_per_cluster: int):
    return (
        jnp.zeros((num_clusters, num_pixels)),  # mnist centroids
        jnp.zeros((num_images_per_cluster, num_pixels)),  # images in cluster 0
        jnp.zeros((num_images_per_cluster, num_pixels)),  # images in cluster 1
        jnp.zeros((num_images_per_cluster, num_pixels)),  # images in cluster 2
        jnp.zeros((num_images_per_cluster, num_pixels)),  # images in cluster 3
        jnp.zeros((num_images_per_cluster, num_pixels)),  # images in cluster 4
        jnp.zeros((num_images_per_cluster, num_pixels)),  # images in cluster 5
        jnp.zeros((num_images_per_cluster, num_pixels)),  # images in cluster 6
        jnp.zeros((num_images_per_cluster, num_pixels)),  # images in cluster 7
        jnp.zeros((num_images_per_cluster, num_pixels)),  # images in cluster 8
        jnp.zeros((num_images_per_cluster, num_pixels)),  # images in cluster 9
        #jnp.zeros((20,param_size)),  # parameter sets
        #jnp.zeros((20,param_size)),  # standard deviations
        #jnp.zeros((20,param_size)),  # scaled noises
        #jnp.full((20,),1000),  # fitness values, adversarial
        #jnp.full((20,),1000),  # fitness values, mutual information
        #jnp.full((20,),1000),  # Tchebyschev fitness values
        #jnp.full((20,),1000),  # entropy
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
    features: int, key: jax.Array, num_codes: int = 10
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
        jnp.ones((pop_size,10,1)),  # pop_avg_spread
        jnp.ones((pop_size,10,10)),  # pop_min_safety

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
    # momentum=0.7 (default): 30% new data blended in each step (fast adaptation)
    # momentum=0.97+: centroids nearly frozen (prevents drift of locked codes)
    updated_hist_avg_per_code = momentum * hist_avg_per_code + (1.0 - momentum) * avg_per_code

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
    domain_ks = belief_space[1]
    situational_ks = belief_space[2]
    history_ks = belief_space[3]
    topographic_ks = belief_space[4]
    normative_ks = belief_space[5]

    best_fitness_variance_ratio = normative_ks[13]
    best_fitness_adv_short_slope = normative_ks[10]
    best_fitness_mi_short_slope = normative_ks[11]
    entropy_short_slope = normative_ks[12]
    stagnation_slope = normative_ks[10]

    best_fitness_adv_med_slope = normative_ks[21]
    best_fitness_mi_med_slope = normative_ks[22]
    entropy_med_slope = normative_ks[23]
    entropy_long_slope = normative_ks[24]

    sit_score = situational_score(best_fitness_adv_short_slope, best_fitness_mi_short_slope)
    hist_score = historical_score(entropy_long_slope, best_fitness_adv_short_slope)
    topo_score = topographic_score(entropy_long_slope, best_fitness_adv_med_slope)
    dom_score = domain_score(best_fitness_adv_med_slope, best_fitness_mi_med_slope, entropy_long_slope)

    ks_weights = jnp.array([dom_score, sit_score, hist_score, topo_score])

    #jax.debug.print('ks weights {} : ', ks_weights)
    #ks_weights = update_ks_weights(
    #    best_fitness_slope,
    #    best_fitness_slope_mi,
    #    norm_entropy_slope,
    #    stagnation_slope,
    #    best_fitness_variance_ratio,
    #)

    min_index = jnp.argmax(ks_weights)
    result = jnp.zeros_like(ks_weights, dtype=jnp.int32)

     
    ks_weights = result.at[min_index].set(1)

    domain_ks_center = domain_ks[0][0]
    situational_ks_center = situational_ks[0] # [:,:n]

    history_max_entropy_idx = jnp.argmax(history_ks[6])
    
    history_ks_center = history_ks[0][history_max_entropy_idx]

    #normative_ks_rolling_digits = normative_ks[5]
    
    # find all unique digits in normative_ks_rolling_digits and order them based on their first occurrence
    #unique_digits = jnp.unique(normative_ks_rolling_digits)

    # find the first digit, 0-9, missing from the unique_digits without for loop
    #missing_digits = jnp.setdiff1d(jnp.arange(10), unique_digits)

    topographic_ks_center = normative_ks[15]
    #domain_ks_center_weighted = domain_ks_center * ks_weights[0]
    domain_ks_center_weighted = domain_ks_center * 0
    situational_row_averages_weighted = situational_ks_center * ks_weights[1]
    history_row_averages_weighted = history_ks_center * ks_weights[2]
    topographic_ks_center_weighted = topographic_ks_center * ks_weights[3]

    return (
        jnp.sum(
            jnp.array(
                [
                    domain_ks_center_weighted +
                    situational_row_averages_weighted +
                    history_row_averages_weighted +
                    topographic_ks_center_weighted
                ]
            ),
            axis=0,
        )
    )

@jax.jit
def get_stdev_guidance(belief_space, t, stdev):
   
    domain_ks = belief_space[1]
    situational_ks = belief_space[2]
    history_ks = belief_space[3]
    topographic_ks = belief_space[4]
    normative_ks = belief_space[5]

    best_fitness_variance_ratio = normative_ks[13]
    best_fitness_adv_short_slope = normative_ks[10]
    best_fitness_mi_short_slope = normative_ks[11]
    entropy_short_slope = normative_ks[12]
    stagnation_slope = normative_ks[10]
 
    best_fitness_adv_med_slope = normative_ks[21]
    best_fitness_mi_med_slope = normative_ks[22]
    entropy_med_slope = normative_ks[23]
    entropy_long_slope = normative_ks[24]

    sit_score = situational_score(best_fitness_adv_short_slope, best_fitness_mi_short_slope)
    hist_score = historical_score(entropy_long_slope, best_fitness_adv_short_slope)
    topo_score = topographic_score(entropy_long_slope, best_fitness_adv_med_slope)
    dom_score = domain_score(best_fitness_adv_med_slope, best_fitness_mi_med_slope, entropy_long_slope)

    ks_weights = jnp.array([dom_score, sit_score, hist_score, topo_score])


    #ks_weights = update_ks_weights(
    #    best_fitness_slope,
    #    best_fitness_slope_mi,
    #    norm_entropy_slope,
    #    stagnation_slope,
    #    best_fitness_variance_ratio,
    #)

    min_index = jnp.argmax(ks_weights)
    result = jnp.zeros_like(ks_weights, dtype=jnp.int32)

    ks_weights = result.at[min_index].set(1)

    domain_ks_stdev = domain_ks[1][0]

    situational_ks_stdev = situational_ks[1]

    history_ks_max_entropy_idx = jnp.argmax(history_ks[6])

    history_ks_stdev = history_ks[1][history_ks_max_entropy_idx]

    normative_ks_rolling_digits = normative_ks[5]

    #unique_digits = jnp.unique(normative_ks_rolling_digits)

    #missing_digits = jnp.setdiff1d(jnp.arange(10), unique_digits)
    #first_missing_digit = missing_digits[0]

    topographic_ks_stdev = normative_ks[16] 

    domain_ks_stdev_weighted = domain_ks_stdev * 0
    situational_row_averages_weighted = situational_ks_stdev * ks_weights[1]
    history_row_averages_weighted = history_ks_stdev * ks_weights[2]

    topographic_ks_stdev = topographic_ks_stdev * ks_weights[3]

    return (jnp.sum(
        jnp.array([
        domain_ks_stdev_weighted
        + situational_row_averages_weighted
        + history_row_averages_weighted
        + topographic_ks_stdev
        ]
        ), axis=0
    )), min_index
