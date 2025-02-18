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


def initialize_domain_ks(param_size: int):
    return (
        jnp.zeros((20,param_size)),  # parameter sets
        jnp.zeros((20,param_size)),  # standard deviations
        jnp.zeros((20,param_size)),  # scaled noises
        jnp.full((20,),1000),  # fitness values, adversarial
        jnp.full((20,),1000),  # fitness values, mutual information
        jnp.full((20,),1000),  # Tchebyschev fitness values
        jnp.full((20,),1000),  # entropy
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
    param_size: int, max_individuals: int = 2
):
    return (
        jnp.zeros((max_individuals, param_size)),  # best solutions for 0
        jnp.zeros((max_individuals, param_size)),  # best stdev for 0
        jnp.zeros((max_individuals, param_size)),  # best scaled noise for 0
        jnp.full((max_individuals,),1000),  # fitness values, adversarial for 0
        jnp.full((max_individuals,),1000),  # fitness values, mutual information for 0
        jnp.full((max_individuals,),1000),  # q value for 0

        jnp.zeros((max_individuals, param_size)), # best solutions for 1
        jnp.zeros((max_individuals, param_size)),  # best stdev for 1
        jnp.zeros((max_individuals, param_size)),  # best scaled noise for 1
        jnp.full((max_individuals,),1000),  # fitness values, adversarial for 1
        jnp.full((max_individuals,),1000),  # fitness values, mutual information for 1
        jnp.full((max_individuals,),1000),  # q value for 1

        jnp.zeros((max_individuals, param_size)), # best solutions for 2
        jnp.zeros((max_individuals, param_size)),  # best stdev for 2
        jnp.zeros((max_individuals, param_size)),  # best scaled noise for 2
        jnp.full((max_individuals,),1000),  # fitness values, adversarial for 2
        jnp.full((max_individuals,),1000),  # fitness values, mutual information for 2
        jnp.full((max_individuals,),1000),  # q value for 2

        jnp.zeros((max_individuals, param_size)), # best solutions for 3
        jnp.zeros((max_individuals, param_size)),  # best stdev for 3
        jnp.zeros((max_individuals, param_size)),  # best scaled noise for 3
        jnp.full((max_individuals,),1000),  # fitness values, adversarial for 3
        jnp.full((max_individuals,),1000),  # fitness values, mutual information for 3
        jnp.full((max_individuals,),1000),  # q value for 3

        jnp.zeros((max_individuals, param_size)), # best solutions for 4
        jnp.zeros((max_individuals, param_size)),  # best stdev for 4
        jnp.zeros((max_individuals, param_size)),  # best scaled noise for 4
        jnp.full((max_individuals,),1000),  # fitness values, adversarial for 4
        jnp.full((max_individuals,),1000),  # fitness values, mutual information for 4
        jnp.full((max_individuals,),1000),  # q value for 4
        
        jnp.zeros((max_individuals, param_size)), # best solutions for 5
        jnp.zeros((max_individuals, param_size)),  # best stdev for 5
        jnp.zeros((max_individuals, param_size)),  # best scaled noise for 5
        jnp.full((max_individuals,),1000),  # fitness values, adversarial for 5
        jnp.full((max_individuals,),1000),  # fitness values, mutual information for 5
        jnp.full((max_individuals,),1000),  # q value for 5

        jnp.zeros((max_individuals, param_size)), # best solutions for 6
        jnp.zeros((max_individuals, param_size)),  # best stdev for 6
        jnp.zeros((max_individuals, param_size)),  # best scaled noise for 6
        jnp.full((max_individuals,),1000),  # fitness values, adversarial for 6
        jnp.full((max_individuals,),1000),  # fitness values, mutual information for 6
        jnp.full((max_individuals,),1000),  # q value for 6

        jnp.zeros((max_individuals, param_size)), # best solutions for 7
        jnp.zeros((max_individuals, param_size)),  # best stdev for 7
        jnp.zeros((max_individuals, param_size)),  # best scaled noise for 7
        jnp.full((max_individuals,),1000),  # fitness values, adversarial for 7
        jnp.full((max_individuals,),1000),  # fitness values, mutual information for 7
        jnp.full((max_individuals,),1000),  # q value for 7

        jnp.zeros((max_individuals, param_size)), # best solutions for 8
        jnp.zeros((max_individuals, param_size)),  # best stdev for 8
        jnp.zeros((max_individuals, param_size)),  # best scaled noise for 8
        jnp.full((max_individuals,),1000),  # fitness values, adversarial for 8
        jnp.full((max_individuals,),1000),  # fitness values, mutual information for 8
        jnp.full((max_individuals,),1000),  # q value for 8

        jnp.zeros((max_individuals, param_size)), # best solutions for 9
        jnp.zeros((max_individuals, param_size)),  # best stdev for 9
        jnp.zeros((max_individuals, param_size)),  # best scaled noise for 9
        jnp.full((max_individuals,),1000),  # fitness values, adversarial for 9
        jnp.full((max_individuals,),1000),  # fitness values, mutual information for 9
        jnp.full((max_individuals,),1000),  # q value for 9

    )


def initialize_normative_ks(param_size: int, pop_size: int = 64):
    return (
        jnp.ones(5),  # rolling_best_fitness_adv_short
        jnp.ones(5),  # rolling_best_fitness_mi_short
        jnp.ones(5),  # rolling_avg_fitness_adv_short
        jnp.ones(5),  # rolling_avg_fitness_mi_short
        jnp.ones(5),  # rolling_rng_adv_short
        jnp.ones(60),  # rolling_digits
        jnp.ones(5),  # rolling_best_tchebyschev_scores_short
        jnp.ones(5),  # rolling_entropy_short
        jnp.array([0.0]),  # best_fitness_adv
        jnp.array([0.0]),  # best_fitness_mi
        jnp.array([0.0]),  # best_fitness_slope_adv
        jnp.array([0.0]),  # best_fitness_slope_mi
        jnp.array([0.0]),  # norm_entropy_slope
        jnp.array([0.0]),  # best_fitness_var_ratio
        jnp.array([0.0]),  # missing_digit
        jnp.zeros((1, param_size)),  # topographic best solution
        jnp.zeros((1, param_size)),  # topographic best stdev
        jnp.ones(40), # rolling_best_fitness_slope_adv_med
        jnp.ones(40), # rolling_best_fitness_slope_mi_med
        jnp.ones(40), # rolling_entropy_med
        jnp.ones(100), # rolling_entropy_long
        jnp.array([0.0]),  # best_fitness_slope_adv_med
        jnp.array([0.0]),  # best_fitness_slope_mi_med
        jnp.array([0.0]),  # norm_entropy_slope_med
        jnp.array([0.0]),  # norm_entropy_slope_long

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
def update_topographic_ks_idx_zero(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, disc_logit, max_individuals=2
):
    topographic_ks = belief_space[4]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, entropies = topographic_ks[0:6]

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
   
    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))

    num_selected = max_individuals

    selected_indices = order[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_topographic_ks_digit = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_entropy,
    )
    
  
    updated_topographic_ks_zero = updated_topographic_ks_digit + topographic_ks[6:60]
    
    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks_zero,) + belief_space[5:] 
    )
    
    return updated_belief_space_topographic

@jax.jit
def update_topographic_ks_idx_one(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, disc_logit, max_individuals=2
):
    topographic_ks = belief_space[4]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, entropies = topographic_ks[6:12]

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
   
    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))

    num_selected = max_individuals

    selected_indices = order[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_topographic_ks_digit_one = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_entropy,
    )
   
    updated_topographic_ks_one = topographic_ks[:6] + updated_topographic_ks_digit_one + topographic_ks[12:60]
    
    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks_one,) + belief_space[5:]
    )
    return updated_belief_space_topographic

@jax.jit
def update_topographic_ks_idx_two(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, disc_logit, max_individuals=2
):
    topographic_ks = belief_space[4]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, entropies = topographic_ks[12:18]

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
   
    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))

    num_selected = max_individuals

    selected_indices = order[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_topographic_ks_digit_two = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_entropy,
    )
    
    updated_topographic_ks_two = topographic_ks[:12] + updated_topographic_ks_digit_two + topographic_ks[18:60] 

    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks_two,) + belief_space[5:]
    )
    return updated_belief_space_topographic

@jax.jit
def update_topographic_ks_idx_three(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, disc_logit, max_individuals=2
):
    topographic_ks = belief_space[4]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, entropies = topographic_ks[18:24]

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
   
    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))

    num_selected = max_individuals

    selected_indices = order[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_topographic_ks_digit_three = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_entropy,
    )
  
    updated_topographic_ks_three = topographic_ks[:18] + updated_topographic_ks_digit_three + topographic_ks[24:60]
    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks_three,) + belief_space[5:]
    )
    return updated_belief_space_topographic

@jax.jit
def update_topographic_ks_idx_four(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, disc_logit, max_individuals=2
):
    topographic_ks = belief_space[4]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, entropies = topographic_ks[24:30]

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
   
    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))

    num_selected = max_individuals

    selected_indices = order[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_topographic_ks_digit_four = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_entropy,
    )
   
    updated_topographic_ks_four = topographic_ks[:24] + updated_topographic_ks_digit_four + topographic_ks[30:60]
    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks_four,) + belief_space[5:]
    )
    return updated_belief_space_topographic

@jax.jit
def update_topographic_ks_idx_five(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, disc_logit, max_individuals=2
):
    topographic_ks = belief_space[4]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, entropies = topographic_ks[30:36]

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
   
    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))

    num_selected = max_individuals

    selected_indices = order[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_topographic_ks_digit_five = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_entropy,
    )
   
    updated_topographic_ks_five = topographic_ks[:30] + updated_topographic_ks_digit_five + topographic_ks[36:60]
    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks_five,) + belief_space[5:]
    )
    return updated_belief_space_topographic

@jax.jit
def update_topographic_ks_idx_six(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, disc_logit, max_individuals=2
):
    topographic_ks = belief_space[4]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, entropies = topographic_ks[36:42]

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
   
    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))

    num_selected = max_individuals

    selected_indices = order[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_topographic_ks_digit_six = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_entropy,
    )
   
    updated_topographic_ks_six = topographic_ks[:36] + updated_topographic_ks_digit_six + topographic_ks[42:60]
    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks_six,) + belief_space[5:]
    )
    return updated_belief_space_topographic

@jax.jit
def update_topographic_ks_idx_seven(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, disc_logit, max_individuals=2
):
    topographic_ks = belief_space[4]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, entropies = topographic_ks[42:48]

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
   
    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))

    num_selected = max_individuals

    selected_indices = order[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_topographic_ks_digit_seven = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_entropy,
    )
   
    updated_topographic_ks_seven = topographic_ks[:42] + updated_topographic_ks_digit_seven + topographic_ks[48:60]
    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks_seven,) + belief_space[5:]
    )
    return updated_belief_space_topographic

@jax.jit
def update_topographic_ks_idx_eight(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, disc_logit, max_individuals=2
):
    topographic_ks = belief_space[4]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, entropies = topographic_ks[48:54]

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
   
    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))

    num_selected = max_individuals

    selected_indices = order[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_topographic_ks_digit_eight = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_entropy,
    )
   
    updated_topographic_ks_eight = topographic_ks[:48] + updated_topographic_ks_digit_eight + topographic_ks[54:60]
    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks_eight,) + belief_space[5:]
    )
    return updated_belief_space_topographic

@jax.jit
def update_topographic_ks_idx_nine(
    belief_space, best_solution, stdev, scaled_noise, fitness_value_adv, fitness_value_mi, disc_logit, max_individuals=2
):
    topographic_ks = belief_space[4]

    best_solutions, best_stdevs, best_scaled_noises, best_fitnesses_adversarial, best_fitnesses_mutual_info, entropies = topographic_ks[54:60]

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
   
    entropy = jnp.array([jnp.sum(-jnp.log(disc_logit + 1e-8) * disc_logit)])

    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )

    objectives = jnp.stack([abs(updated_best_fitnesses_adversarial), abs(updated_best_fitnesses_mutual_info), updated_entropy], axis=1)

    ranks = non_dominated_sort_lax(objectives)

    order = jnp.lexsort((-updated_best_fitnesses_adversarial, ranks))

    num_selected = max_individuals

    selected_indices = order[:num_selected]

    selected_best_solutions = updated_best_solutions[selected_indices]
    selected_best_stdevs = updated_best_stdevs[selected_indices]
    selected_best_scaled_noises = updated_best_scaled_noises[selected_indices]
    selected_best_fitnesses_adversarial = updated_best_fitnesses_adversarial[selected_indices]
    selected_best_fitnesses_mutual_info = updated_best_fitnesses_mutual_info[selected_indices]
    selected_entropy = updated_entropy[selected_indices]

    updated_topographic_ks_digit_nine = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_entropy,
    )
   
    updated_topographic_ks_nine = topographic_ks[:54] + updated_topographic_ks_digit_nine
    updated_belief_space_topographic = (
        belief_space[:4] + (updated_topographic_ks_nine,) + belief_space[5:]
    )
    return updated_belief_space_topographic

@jax.jit
def update_normative_ks(
    belief_space, best_fitness, best_fitness_mi, avg_fitness, avg_fitness_mi, best_adv, best_mi, rng_adv, digit, best_tchebycheff_scores, softmax_logits, missing_digit, topographic_center, topographic_stdev
):
    normative_ks = belief_space[5]

    one_dim_avg_fitness = jnp.array([avg_fitness])
    one_dim_best_fitness = jnp.array([best_fitness])
    one_dim_best_fitness_mi = jnp.array([best_fitness_mi])
    one_dim_avg_fitness_mi = jnp.array([avg_fitness_mi])
    one_dim_rng_adv = jnp.array([rng_adv])
    one_dim_digit = jnp.array([digit])
    one_dim_best_tchebycheff_scores = jnp.array([best_tchebycheff_scores])

    entropy = jnp.array([jnp.sum(-jnp.log(softmax_logits + 1e-8) * softmax_logits)])
    #one_dim_norm_entropy = jnp.array([norm_entropy])
    
    updated_rolling_best_fitness = jnp.concatenate(
        [normative_ks[0], one_dim_best_fitness], axis=0
    )[1:]
  
    updated_rolling_best_fitness_mi = jnp.concatenate(
        [normative_ks[1], one_dim_best_fitness_mi], axis=0
    )[1:]
    
    updated_rolling_avg_fitness = jnp.concatenate(
        [normative_ks[2], one_dim_avg_fitness], axis=0
    )[1:]

    updated_rolling_avg_fitness_mi = jnp.concatenate(
        [normative_ks[3], one_dim_avg_fitness_mi], axis=0
    )[1:]

    updated_rolling_rng_adv = jnp.concatenate(
        [normative_ks[4], one_dim_rng_adv], axis=0
    )[1:]

    updated_rolling_digit = jnp.concatenate(
        [normative_ks[5], one_dim_digit], axis=0
    )[1:]

    updated_rolling_best_tchebyscheff_scores = jnp.concatenate(
        [normative_ks[6], one_dim_best_tchebycheff_scores], axis=0
    )[1:]

    updated_rolling_entropy = jnp.concatenate(
        [normative_ks[7], entropy], axis=0
    )[1:]
    
    updated_rolling_best_fitness_med = jnp.concatenate(
        [normative_ks[17], one_dim_best_fitness], axis=0
    )[1:]

    updated_rolling_best_fitness_mi_med = jnp.concatenate(
        [normative_ks[18], one_dim_best_fitness_mi], axis=0
    )[1:]

    updated_rolling_entropy_med = jnp.concatenate(
        [normative_ks[19], entropy], axis=0
    )[1:]

    updated_rolling_entropy_long = jnp.concatenate(
        [normative_ks[20], entropy], axis=0
    )[1:]

    #updated_rolling_norm_entropy = jnp.concatenate(
    #    [normative_ks[2], one_dim_norm_entropy], axis=0
    #)[1:]

    (
        scaled_rolling_best_fitness,
        scaled_rolling_best_fitness_mi,
        scaled_rolling_entropy,
    ) = scale_arrays(
        [
            updated_rolling_best_fitness,
            updated_rolling_best_fitness_mi,
            updated_rolling_entropy,
        ]
    )

    scaled_rolling_best_fitness_variance = jnp.var(scaled_rolling_best_fitness)
    scaled_rolling_best_fitness_mi_variance = jnp.var(scaled_rolling_best_fitness_mi)

    one_dim_best_fitness_variance = jnp.array([scaled_rolling_best_fitness_variance])
    one_dim_best_fitness_mi_variance = jnp.array([scaled_rolling_best_fitness_mi_variance])

    updated_rolling_best_fitness_variance = jnp.concatenate(
        [normative_ks[7], one_dim_best_fitness_variance], axis=0
    )[1:]

    top_20_variances = jax.lax.top_k(updated_rolling_best_fitness_variance, 5)[0]

    average_rolling_best_fitness_variance = jnp.mean(top_20_variances)

    best_fitness_variance_ratio = (
        scaled_rolling_best_fitness_variance / average_rolling_best_fitness_variance
    )

    #best_fitness_slope, best_fitness_slope_mi, entropy_slope, stagnation_slope = (
    #    calculate_slopes(
    #        best_fitness_window=scaled_rolling_best_fitness,
    #        best_fitness_window_mi=scaled_rolling_best_fitness_mi,
    #        norm_entropy_window=scaled_rolling_entropy,
    #    )
    #)

    best_fitness_adv_short_slope = calculate_slope(updated_rolling_best_fitness)
    best_fitness_mi_short_slope = calculate_slope(updated_rolling_best_fitness_mi)
    entropy_slope_short = calculate_slope(updated_rolling_entropy)

    best_fitness_adv_med_slope = calculate_slope(updated_rolling_best_fitness_med)
    best_fitness_mi_med_slope = calculate_slope(updated_rolling_best_fitness_mi_med)
    entropy_slope_med = calculate_slope(updated_rolling_entropy_med)
    entropy_slope_long = calculate_slope(updated_rolling_entropy_long)

    #ks_weights = update_ks_weights(
    #    best_fitness_slope,
    #    best_fitness_slope_mi,
    #    entropy_slope,
    #    stagnation_slope,
    #    best_fitness_variance_ratio,
    #)

    updated_normative_ks = (
        updated_rolling_best_fitness,
        updated_rolling_best_fitness_mi,
        updated_rolling_avg_fitness,
        updated_rolling_avg_fitness_mi,
        updated_rolling_rng_adv,
        updated_rolling_digit,
        updated_rolling_best_tchebyscheff_scores,
        updated_rolling_entropy,
        best_adv,
        best_mi,
        best_fitness_adv_short_slope,
        best_fitness_mi_short_slope,
        entropy_slope_short,
        best_fitness_variance_ratio,
        missing_digit,
        topographic_center,
        topographic_stdev,
        updated_rolling_best_fitness_med,
        updated_rolling_best_fitness_mi_med,
        updated_rolling_entropy_med,
        updated_rolling_entropy_long,
        best_fitness_adv_med_slope,
        best_fitness_mi_med_slope,
        entropy_slope_med,
        entropy_slope_long,
    )

    updated_belief_space_normative = (
        belief_space[:5] + (updated_normative_ks,)
    )

    return updated_belief_space_normative#, ks_weights

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
