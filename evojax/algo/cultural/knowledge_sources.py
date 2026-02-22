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


def initialize_metric_history(window_size: int = 100):
    """Rolling window for computing metric slopes across generations.

    Tracks key metrics per generation for CA-driven weight adaptation.
    Slopes are computed over short (last 20), medium (last 50), and long
    (full window) horizons to detect stagnation, mode collapse, divergence,
    and improvement trends.
    """
    return (
        jnp.zeros((window_size,)),   # [0] best_fitness_adv per generation
        jnp.zeros((window_size,)),   # [1] best_fitness_mi per generation
        jnp.zeros((window_size,)),   # [2] entropy per generation
        jnp.zeros((window_size,)),   # [3] best_r_sense per generation
        jnp.int32(0),                # [4] write_index (circular buffer position)
        jnp.int32(0),                # [5] count (entries written, capped at window_size)
        jnp.zeros((window_size,)),   # [6] avg_r_intra per generation
        jnp.zeros((window_size,)),   # [7] avg_fitness_adv per generation
    )


def initialize_control_ks():
    """Runtime control state for CA-driven adaptive training knobs.

    State layout:
      [0] ema_signals: [d_dominance, shortcut_risk, diversity_distress, stagnation, prototype_lock]
      [1] adaptive_weights: [w_adv, w_mi, w_div, w_sense, w_intra, w_cons_floor, w_norm, w_adv_ceiling]
      [2] adaptive_adv_ceiling
      [3] adaptive_ca_blend
      [4] d_update_rate  (0..1, higher = more D updates)
    """
    return (
        jnp.zeros((5,), dtype=jnp.float32),
        jnp.zeros((8,), dtype=jnp.float32),
        jnp.float32(-0.65),
        jnp.float32(0.0),
        jnp.float32(0.5),
    )


def initialize_domain_ks(param_size: int, num_elites: int = 20):
    return (
        jnp.zeros((num_elites, param_size)),  # [0] best solutions (parameter sets)
        jnp.zeros((num_elites, param_size)),  # [1] stdevs
        jnp.zeros((num_elites, param_size)),  # [2] scaled noises
        jnp.full((num_elites,), 1000.0),      # [3] fitness values, adversarial
        jnp.full((num_elites,), 1000.0),      # [4] fitness values, mutual information
        jnp.full((num_elites,), 1000.0),      # [5] combined fitness values
        jnp.full((num_elites,), 1000.0),      # [6] entropy
        jnp.zeros((num_elites,)),             # [7] r_sense (code separation quality)
        jnp.zeros((num_elites,)),             # [8] r_cons (constraint satisfaction)
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
def update_metric_history(belief_space, best_fitness_adv, best_fitness_mi, entropy, best_r_sense, avg_r_intra=0.0, avg_fitness_adv=0.0):
    """Append one generation's key metrics to the circular buffer.

    Args:
        belief_space: full belief space tuple (metric_history is element [6])
        best_fitness_adv: scalar, best adversarial fitness this generation
        best_fitness_mi: scalar, best MI fitness this generation
        entropy: scalar, population-level entropy this generation
        best_r_sense: scalar, best code separation this generation
        avg_r_intra: scalar, population average within-code variation
        avg_fitness_adv: scalar, population average adversarial fitness
    """
    metric_history = belief_space[6]
    adv_buf, mi_buf, ent_buf, sense_buf, write_idx, count, intra_buf, adv_avg_buf = metric_history
    window_size = adv_buf.shape[0]

    adv_buf = adv_buf.at[write_idx].set(best_fitness_adv)
    mi_buf = mi_buf.at[write_idx].set(best_fitness_mi)
    ent_buf = ent_buf.at[write_idx].set(entropy)
    sense_buf = sense_buf.at[write_idx].set(best_r_sense)
    intra_buf = intra_buf.at[write_idx].set(avg_r_intra)
    adv_avg_buf = adv_avg_buf.at[write_idx].set(avg_fitness_adv)
    new_idx = (write_idx + 1) % window_size
    new_count = jnp.minimum(count + 1, window_size)

    updated_metric_history = (adv_buf, mi_buf, ent_buf, sense_buf, new_idx, new_count, intra_buf, adv_avg_buf)
    updated_belief_space = belief_space[:6] + (updated_metric_history,) + belief_space[7:]
    return updated_belief_space


@jax.jit
def update_control_ks(
    belief_space,
    t,
    fitness_adv,
    fitness_mi,
    pop_var,
    r_sense,
    r_intra,
    avg_spread,
    base_weights,
    base_adv_ceiling,
    ca_activation_iter,
    ca_blend_start_iter,
    ca_blend_ramp_iters,
    ca_blend_max,
):
    """Update CA runtime control state from metric history + current population.

    This turns the CA belief space into a controller that adapts training
    knobs (fitness weights, adversarial ceiling, CA blend, and D update rate)
    as the GAN landscape changes.
    """
    control_ks = belief_space[7]
    ema_prev, _, _, _, _ = control_ks

    slopes = compute_metric_slopes(belief_space)
    adv_short = slopes[0]
    adv_med = slopes[2]
    mi_med = slopes[3]
    sense_short = slopes[5]
    intra_short = slopes[6]

    best_adv = jnp.max(fitness_adv)
    pop_var_mean = jnp.mean(pop_var)

    # 1) Risk detection.
    # D-dominance: adversarial reward too negative and still trending down.
    d_dom_level = jnp.clip((-1.15 - best_adv) / 0.55, 0.0, 1.0)
    d_dom_trend = jnp.clip((-adv_short) / 0.03, 0.0, 1.0)
    d_dominance_raw = 0.70 * d_dom_level + 0.30 * d_dom_trend

    # Shortcut risk: adversarial reward jumps upward (toward 0) and exceeds ceiling.
    shortcut_level = jnp.clip(
        (best_adv - (base_adv_ceiling - 0.10)) / 0.35, 0.0, 1.0
    )
    shortcut_frac = jnp.mean((fitness_adv > base_adv_ceiling).astype(jnp.float32))
    shortcut_trend = jnp.clip(adv_short / 0.03, 0.0, 1.0)
    shortcut_raw = 0.45 * shortcut_level + 0.35 * shortcut_frac + 0.20 * shortcut_trend

    # Prototype-lock risk: codes are well separated, but each code collapses
    # to near-identical samples across z (very low spread / within-code variance).
    r_sense_mean = jnp.mean(r_sense)
    r_intra_mean = jnp.mean(r_intra)
    spread_deficit = jnp.clip((0.055 - avg_spread) / 0.03, 0.0, 1.0)
    intra_deficit = jnp.clip((0.75 - r_intra_mean) / 0.30, 0.0, 1.0)
    sense_excess = jnp.clip((r_sense_mean - 0.09) / 0.04, 0.0, 1.0)
    prototype_lock_raw = jnp.clip(
        0.45 * intra_deficit + 0.30 * spread_deficit + 0.25 * sense_excess,
        0.0,
        1.0,
    )

    # Diversity distress: code-separation/within-code metrics weakening.
    sense_drop = jnp.clip((-sense_short) / 0.004, 0.0, 1.0)
    intra_drop = jnp.clip((-intra_short) / 0.0025, 0.0, 1.0)
    var_drop = jnp.clip((0.10 - pop_var_mean) / 0.10, 0.0, 1.0)
    diversity_raw = jnp.clip(
        0.30 * sense_drop + 0.25 * intra_drop + 0.15 * var_drop
        + 0.20 * intra_deficit + 0.10 * spread_deficit,
        0.0,
        1.0,
    )

    # Stagnation: medium-term slopes are flat.
    stagnation_raw = 1.0 - jnp.clip(
        (jnp.abs(adv_med) + jnp.abs(mi_med)) / 0.01, 0.0, 1.0
    )

    raw_signals = jnp.array(
        [d_dominance_raw, shortcut_raw, diversity_raw, stagnation_raw, prototype_lock_raw],
        dtype=jnp.float32
    )

    # 2) EMA smoothing for stability.
    ema_alpha = jnp.float32(0.15)
    ema_signals = (1.0 - ema_alpha) * ema_prev + ema_alpha * raw_signals

    # 3) Ramp in control authority after warm-up.
    control_ramp = jnp.clip((t - ca_activation_iter) / 2000.0, 0.0, 1.0)
    d_dominance, shortcut_risk, diversity_distress, stagnation, prototype_lock = ema_signals * control_ramp

    # 4) Adaptive fitness weights.
    base_w_adv = base_weights[0]
    base_w_mi = base_weights[1]
    base_w_div = base_weights[2]
    base_w_sense = base_weights[3]
    base_w_intra = base_weights[4]
    base_w_cons_floor = base_weights[5]
    base_w_norm = base_weights[6]
    base_w_adv_ceiling = base_weights[7]

    w_adv = base_w_adv + 0.14 * d_dominance - 0.16 * shortcut_risk
    w_mi = base_w_mi + 0.03 * stagnation
    w_div = base_w_div + 0.10 * diversity_distress + 0.10 * prototype_lock - 0.05 * d_dominance
    w_sense = base_w_sense + 0.08 * diversity_distress - 0.10 * prototype_lock - 0.04 * shortcut_risk
    w_intra = base_w_intra + 0.20 * diversity_distress + 0.18 * prototype_lock + 0.06 * stagnation
    w_cons_floor = base_w_cons_floor + 0.06 * shortcut_risk
    w_norm = base_w_norm + 0.07 * shortcut_risk + 0.03 * diversity_distress + 0.02 * prototype_lock
    w_adv_ceiling = base_w_adv_ceiling + 0.90 * shortcut_risk

    adaptive_weights = jnp.array([
        jnp.clip(w_adv, 0.25, 0.80),
        jnp.clip(w_mi, 0.05, 0.25),
        jnp.clip(w_div, 0.20, 0.70),
        jnp.clip(w_sense, 0.03, 0.30),
        jnp.clip(w_intra, 0.05, 0.45),
        jnp.clip(w_cons_floor, 0.02, 0.20),
        jnp.clip(w_norm, 0.0, 0.25),
        jnp.clip(w_adv_ceiling, 0.0, 2.0),
    ], dtype=jnp.float32)

    # 5) Adaptive adversarial ceiling and CA blend strength.
    adaptive_adv_ceiling = jnp.clip(
        base_adv_ceiling + 0.05 * d_dominance - 0.08 * shortcut_risk,
        -0.95, -0.45
    )

    blend_ramp = jnp.clip(
        (t - ca_blend_start_iter) / jnp.maximum(ca_blend_ramp_iters, 1.0),
        0.0, 1.0
    )
    blend_multiplier = jnp.clip(
        0.5 + 0.8 * shortcut_risk + 0.4 * stagnation - 0.3 * d_dominance + 0.3 * prototype_lock,
        0.30, 1.50
    )
    adaptive_ca_blend = ca_blend_max * blend_ramp * blend_multiplier

    # 6) D update control (trainer uses this to modulate D frequency).
    d_update_rate = jnp.clip(
        0.45 + 0.45 * shortcut_risk - 0.85 * d_dominance - 0.25 * prototype_lock + 0.10 * stagnation,
        0.0, 1.0
    )
    d_update_rate = jnp.where(control_ramp > 0.0, d_update_rate, jnp.float32(0.5))

    updated_control_ks = (
        ema_signals,
        adaptive_weights,
        adaptive_adv_ceiling,
        adaptive_ca_blend,
        d_update_rate,
    )
    return belief_space[:7] + (updated_control_ks,)


@jax.jit
def get_control_outputs(belief_space):
    """Fetch latest adaptive control outputs from belief space."""
    control_ks = belief_space[7]
    ema_signals, adaptive_weights, adaptive_adv_ceiling, adaptive_ca_blend, d_update_rate = control_ks
    return adaptive_weights, adaptive_adv_ceiling, adaptive_ca_blend, d_update_rate, ema_signals


@jax.jit
def compute_metric_slopes(belief_space):
    """Compute short/medium/long slopes from the metric history circular buffer.

    Returns a tuple of 9 slopes:
        (adv_short, mi_short, adv_med, mi_med, ent_long,
         sense_short, intra_short, adv_avg_short, sense_med)

    Short = last 20 generations, Medium = last 50, Long = full window (100).
    Uses least-squares linear regression.
    Returns 0.0 for any window that doesn't have enough data yet.
    """
    metric_history = belief_space[6]
    adv_buf, mi_buf, ent_buf, sense_buf, write_idx, count, intra_buf, adv_avg_buf = metric_history
    window_size = adv_buf.shape[0]

    def _slope_over_last_n(buf, n, write_idx, count):
        """Compute slope over the last n entries of a circular buffer."""
        has_enough = count >= n
        # Extract the last n entries in chronological order
        indices = (jnp.arange(n) + write_idx - n) % window_size
        vals = buf[indices]
        x = jnp.arange(n, dtype=jnp.float32)
        mean_x = jnp.mean(x)
        mean_y = jnp.mean(vals)
        numer = jnp.sum((x - mean_x) * (vals - mean_y))
        denom = jnp.sum((x - mean_x) ** 2) + 1e-12
        slope = numer / denom
        return jnp.where(has_enough, slope, 0.0)

    adv_short     = _slope_over_last_n(adv_buf, 20, write_idx, count)
    mi_short      = _slope_over_last_n(mi_buf, 20, write_idx, count)
    adv_med       = _slope_over_last_n(adv_buf, 50, write_idx, count)
    mi_med        = _slope_over_last_n(mi_buf, 50, write_idx, count)
    ent_long      = _slope_over_last_n(ent_buf, 100, write_idx, count)
    sense_short   = _slope_over_last_n(sense_buf, 20, write_idx, count)
    intra_short   = _slope_over_last_n(intra_buf, 20, write_idx, count)
    adv_avg_short = _slope_over_last_n(adv_avg_buf, 20, write_idx, count)
    sense_med     = _slope_over_last_n(sense_buf, 50, write_idx, count)

    return (adv_short, mi_short, adv_med, mi_med, ent_long,
            sense_short, intra_short, adv_avg_short, sense_med)


@jax.jit
def update_domain_ks(
    belief_space, best_solution, stdev, best_scaled_noise,
    best_fitness_adv, best_fitness_mi, best_fitness_combined,
    disc_logit, best_r_sense, best_r_cons
):
    """Update Domain KS: Pareto archive with GAN diagnostic metadata.

    Maintains 20 non-dominated solutions ranked by [|adv|, |mi|, |entropy|].
    Each archived solution also tracks r_sense and r_cons for failure-mode
    detection in the adaptive guidance selection.
    """
    domain_ks = belief_space[1]
    (best_solutions, stdevs, best_scaled_noises,
     best_fitnesses_adv, best_fitnesses_mi, best_fitnesses_combined,
     entropies, r_senses, r_conses) = domain_ks

    # Append new solution
    updated_solutions = jnp.concatenate([best_solutions, best_solution], axis=0)
    updated_stdevs = jnp.concatenate([stdevs, stdev.reshape(1, -1)], axis=0)
    updated_noises = jnp.concatenate([best_scaled_noises, best_scaled_noise], axis=0)
    updated_adv = jnp.concatenate([best_fitnesses_adv, best_fitness_adv.flatten()], axis=0)
    updated_mi = jnp.concatenate([best_fitnesses_mi, best_fitness_mi.flatten()], axis=0)
    updated_combined = jnp.concatenate([best_fitnesses_combined, best_fitness_combined.flatten()], axis=0)

    safe_logit = jnp.maximum(disc_logit, 1e-8)
    entropy = jnp.array([jnp.sum(-jnp.log(safe_logit) * safe_logit)])
    updated_entropy = jnp.concatenate([entropies, entropy], axis=0)

    updated_r_sense = jnp.concatenate([r_senses, best_r_sense.flatten()], axis=0)
    updated_r_cons = jnp.concatenate([r_conses, best_r_cons.flatten()], axis=0)

    # Non-dominated sort on [|adv|, |mi|, |entropy|]
    objectives = jnp.stack([
        jnp.abs(updated_adv),
        jnp.abs(updated_mi),
        jnp.abs(updated_entropy)
    ], axis=1)
    ranks = non_dominated_sort_lax(objectives)

    # Select top 20 by Pareto rank, tie-break by adversarial fitness
    num_selected = 20
    order = jnp.lexsort((-updated_adv, ranks))
    selected = order[:num_selected]

    updated_domain_ks = (
        updated_solutions[selected],
        updated_stdevs[selected],
        updated_noises[selected],
        updated_adv[selected],
        updated_mi[selected],
        updated_combined[selected],
        updated_entropy[selected],
        updated_r_sense[selected],
        updated_r_cons[selected],
    )

    return belief_space[:1] + (updated_domain_ks,) + belief_space[2:]

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

    safe_logit = jnp.maximum(disc_logit, 1e-8)
    entropy = jnp.array([jnp.sum(-jnp.log(safe_logit) * safe_logit)])

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

    safe_logit = jnp.maximum(disc_logit, 1e-8)
    entropy = jnp.array([jnp.sum(-jnp.log(safe_logit) * safe_logit)])

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
        belief_space[:5] + (updated_normative_ks,) + belief_space[6:]
    )

    return updated_belief_space_normative

@jax.jit
def _domain_ks_select_index(domain_ks, entropy_long_slope, adv_med_slope):
    """Adaptive buffer: select which archived solution Domain KS suggests.

    The Domain KS embodies knowledge about GAN training dynamics.  Instead of
    always returning the first Pareto solution, it detects the current training
    regime and picks the most appropriate archived solution:

    - Mode collapse risk (entropy dropping): pick highest-entropy solution
      to recover diversity.
    - Stagnation (adv not improving): pick highest r_sense solution to
      explore code separation — a different axis of improvement.
    - Normal progress: pick solution with best combined fitness (exploit
      the domain's governing rules).
    """
    entropies = domain_ks[6]     # (20,)
    r_senses = domain_ks[7]      # (20,)
    combined = domain_ks[5]      # (20,)

    # Detect mode collapse: entropy slope is negative (dropping)
    collapse_risk = entropy_long_slope < -0.005

    # Detect stagnation: adversarial slope is near zero (no medium-term improvement)
    stagnation = jnp.abs(adv_med_slope) < 0.001

    # Select index based on detected regime
    # Priority: collapse > stagnation > normal
    # (collapse is the most dangerous failure mode)
    idx_entropy = jnp.argmax(entropies)      # highest entropy (recover diversity)
    idx_sense = jnp.argmax(r_senses)         # best separation (explore new axis)
    idx_combined = jnp.argmax(combined)      # best overall (exploit)

    idx = jnp.where(collapse_risk, idx_entropy,
          jnp.where(stagnation, idx_sense, idx_combined))
    return idx


@jax.jit
def get_center_guidance(belief_space, t, center):
    domain_ks = belief_space[1]
    situational_ks = belief_space[2]
    history_ks = belief_space[3]
    topographic_ks = belief_space[4]
    normative_ks = belief_space[5]

    # Compute slopes from metric history
    slopes = compute_metric_slopes(belief_space)
    adv_short, mi_short, adv_med, mi_med, ent_long = slopes[:5]

    # Score each KS based on current training dynamics
    sit_score = situational_score(adv_short, mi_short)
    hist_score = historical_score(ent_long, adv_short)
    topo_score = topographic_score(ent_long, adv_med)
    dom_score = domain_score(adv_med, mi_med, ent_long)

    ks_scores = jnp.array([dom_score, sit_score, hist_score, topo_score])

    # CATGAME-inspired weighted distribution: all KS contribute proportionally
    # rather than winner-take-all.  Temperature controls diversity —
    # higher values spread influence more uniformly across KS, preventing
    # one KS from dominating with oscillation (the WTD failure mode).
    # At temperature=2.0, a score difference of 1.0 yields ~60/40 split
    # instead of 100/0, achieving the dynamic equilibrium of CATGAME.
    temperature = 2.0
    ks_weights = jax.nn.softmax(ks_scores / temperature)

    # Domain KS: adaptive buffer selection (failure-mode aware)
    domain_idx = _domain_ks_select_index(domain_ks, ent_long, adv_med)
    domain_ks_center = domain_ks[0][domain_idx]

    # Situational KS: current best (most exploitative)
    situational_ks_center = situational_ks[0].flatten()

    # Historical KS: best historical solution by entropy (recover diversity)
    history_max_entropy_idx = jnp.argmax(history_ks[6])
    history_ks_center = history_ks[0][history_max_entropy_idx]

    # Topographic KS operates in output space (centroids/fitness shaping),
    # not parameter space.  Its parameter-space contribution is conservative:
    # "keep current center."  When topographic score is high (exploring output
    # structure), guidance favors stability over moving toward archived solutions.
    topo_ks_center = center.flatten()

    guidance = (
        domain_ks_center * ks_weights[0]
        + situational_ks_center * ks_weights[1]
        + history_ks_center * ks_weights[2]
        + topo_ks_center * ks_weights[3]
    )

    return guidance


@jax.jit
def get_stdev_guidance(belief_space, t, stdev):
    domain_ks = belief_space[1]
    situational_ks = belief_space[2]
    history_ks = belief_space[3]
    topographic_ks = belief_space[4]
    normative_ks = belief_space[5]

    # Compute slopes from metric history
    slopes = compute_metric_slopes(belief_space)
    adv_short, mi_short, adv_med, mi_med, ent_long = slopes[:5]

    sit_score = situational_score(adv_short, mi_short)
    hist_score = historical_score(ent_long, adv_short)
    topo_score = topographic_score(ent_long, adv_med)
    dom_score = domain_score(adv_med, mi_med, ent_long)

    ks_scores = jnp.array([dom_score, sit_score, hist_score, topo_score])

    # CATGAME-inspired weighted distribution (see get_center_guidance)
    temperature = 2.0
    ks_weights = jax.nn.softmax(ks_scores / temperature)

    # Domain KS: adaptive buffer selection (same failure-mode logic)
    domain_idx = _domain_ks_select_index(domain_ks, ent_long, adv_med)
    domain_ks_stdev = domain_ks[1][domain_idx]

    situational_ks_stdev = situational_ks[1].flatten()

    history_ks_max_entropy_idx = jnp.argmax(history_ks[6])
    history_ks_stdev = history_ks[1][history_ks_max_entropy_idx]

    # Topographic: conservative — keep current exploration rate
    topo_ks_stdev = stdev.flatten()

    guidance = (
        domain_ks_stdev * ks_weights[0]
        + situational_ks_stdev * ks_weights[1]
        + history_ks_stdev * ks_weights[2]
        + topo_ks_stdev * ks_weights[3]
    )

    max_index = jnp.argmax(ks_weights)
    return guidance, max_index
