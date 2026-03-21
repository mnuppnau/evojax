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
        jnp.zeros((window_size,)),   # [8] avg_r_shape_div per generation
        jnp.zeros((window_size,)),   # [9] avg_code_spread per generation
        jnp.zeros((window_size,)),   # [10] avg_morph_dark_range per generation
        jnp.zeros((window_size,)),   # [11] avg_code_proto_corr per generation
        jnp.zeros((window_size,)),   # [12] avg_proto_angle_spread per generation
        jnp.zeros((window_size,)),   # [13] avg_nuc_cell_ratio_range per generation
        jnp.zeros((window_size,)),   # [14] avg_nuc_eccentricity_range per generation
        jnp.zeros((window_size,)),   # [15] avg_cell_circularity per generation
        jnp.zeros((window_size,)),   # [16] avg_cell_area_var per generation
        jnp.zeros((window_size,)),   # [17] avg_nucleus_offset per generation
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
        jnp.zeros((num_elites,)),             # [9] r_shape_div (conditional shape diversity)
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
        jnp.full((num_iterations,),1000),  # entropies
        jnp.zeros((num_iterations,)),      # biology-aware rescue score
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


def initialize_normative_ks(param_size: int, pop_size: int = 64, num_codes: int = 11):
    return (
        jnp.ones((pop_size, num_codes, 1)),  # pop_avg_spread
        jnp.ones((pop_size, num_codes, num_codes)),  # pop_min_safety
        jnp.float32(0.55),  # cell_circularity_floor
        jnp.float32(0.0040),  # cell_area_var_ceiling
        jnp.float32(0.10),  # nucleus_offset_low
        jnp.float32(0.35),  # nucleus_offset_high

    )

@jax.jit
def update_metric_history(
    belief_space,
    best_fitness_adv,
    best_fitness_mi,
    entropy,
    best_r_sense,
    avg_r_intra=0.0,
    avg_fitness_adv=0.0,
    avg_r_shape_div=0.0,
    avg_code_spread=0.0,
    avg_morph_dark_range=0.0,
    avg_code_proto_corr=0.0,
    avg_proto_angle_spread=0.0,
    avg_nuc_cell_ratio_range=0.0,
    avg_nuc_eccentricity_range=0.0,
    avg_cell_circularity=0.0,
    avg_cell_area_var=0.0,
    avg_nucleus_offset=0.0,
):
    """Append one generation's key metrics to the circular buffer.

    Args:
        belief_space: full belief space tuple (metric_history is element [6])
        best_fitness_adv: scalar, best adversarial fitness this generation
        best_fitness_mi: scalar, best MI fitness this generation
        entropy: scalar, population-level entropy this generation
        best_r_sense: scalar, best code separation this generation
        avg_r_intra: scalar, population average within-code variation
        avg_fitness_adv: scalar, population average adversarial fitness
        avg_r_shape_div: scalar, population average conditional shape diversity
        avg_code_spread: scalar, population average between-code pixel separation
        avg_morph_dark_range: scalar, population average morphology dark-range proxy
        avg_code_proto_corr: scalar, population average cross-code prototype correlation
        avg_proto_angle_spread: scalar, population average prototype angle spread
    """
    metric_history = belief_space[6]
    (adv_buf, mi_buf, ent_buf, sense_buf, write_idx, count,
     intra_buf, adv_avg_buf, shape_buf, spread_buf,
     dark_range_buf, code_corr_buf, angle_spread_buf,
     nuc_ratio_buf, nuc_ecc_buf, cell_circ_buf, cell_area_var_buf,
     nuc_offset_buf) = metric_history
    window_size = adv_buf.shape[0]

    # Keep history finite so slope calculations stay valid even if a transient
    # upstream metric overflows/underflows.
    best_fitness_adv = jnp.nan_to_num(best_fitness_adv, nan=0.0, posinf=0.0, neginf=0.0)
    best_fitness_mi = jnp.nan_to_num(best_fitness_mi, nan=0.0, posinf=0.0, neginf=0.0)
    entropy = jnp.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    best_r_sense = jnp.nan_to_num(best_r_sense, nan=0.0, posinf=0.0, neginf=0.0)
    avg_r_intra = jnp.nan_to_num(avg_r_intra, nan=0.0, posinf=0.0, neginf=0.0)
    avg_fitness_adv = jnp.nan_to_num(avg_fitness_adv, nan=0.0, posinf=0.0, neginf=0.0)
    avg_r_shape_div = jnp.nan_to_num(avg_r_shape_div, nan=0.0, posinf=0.0, neginf=0.0)
    avg_code_spread = jnp.nan_to_num(avg_code_spread, nan=0.0, posinf=0.0, neginf=0.0)
    avg_morph_dark_range = jnp.nan_to_num(avg_morph_dark_range, nan=0.0, posinf=0.0, neginf=0.0)
    avg_code_proto_corr = jnp.nan_to_num(avg_code_proto_corr, nan=0.0, posinf=0.0, neginf=0.0)
    avg_proto_angle_spread = jnp.nan_to_num(avg_proto_angle_spread, nan=0.0, posinf=0.0, neginf=0.0)
    avg_nuc_cell_ratio_range = jnp.nan_to_num(avg_nuc_cell_ratio_range, nan=0.0, posinf=0.0, neginf=0.0)
    avg_nuc_eccentricity_range = jnp.nan_to_num(avg_nuc_eccentricity_range, nan=0.0, posinf=0.0, neginf=0.0)
    avg_cell_circularity = jnp.nan_to_num(avg_cell_circularity, nan=0.0, posinf=0.0, neginf=0.0)
    avg_cell_area_var = jnp.nan_to_num(avg_cell_area_var, nan=0.0, posinf=0.0, neginf=0.0)
    avg_nucleus_offset = jnp.nan_to_num(avg_nucleus_offset, nan=0.0, posinf=0.0, neginf=0.0)

    adv_buf = adv_buf.at[write_idx].set(best_fitness_adv)
    mi_buf = mi_buf.at[write_idx].set(best_fitness_mi)
    ent_buf = ent_buf.at[write_idx].set(entropy)
    sense_buf = sense_buf.at[write_idx].set(best_r_sense)
    intra_buf = intra_buf.at[write_idx].set(avg_r_intra)
    adv_avg_buf = adv_avg_buf.at[write_idx].set(avg_fitness_adv)
    shape_buf = shape_buf.at[write_idx].set(avg_r_shape_div)
    spread_buf = spread_buf.at[write_idx].set(avg_code_spread)
    dark_range_buf = dark_range_buf.at[write_idx].set(avg_morph_dark_range)
    code_corr_buf = code_corr_buf.at[write_idx].set(avg_code_proto_corr)
    angle_spread_buf = angle_spread_buf.at[write_idx].set(avg_proto_angle_spread)
    nuc_ratio_buf = nuc_ratio_buf.at[write_idx].set(avg_nuc_cell_ratio_range)
    nuc_ecc_buf = nuc_ecc_buf.at[write_idx].set(avg_nuc_eccentricity_range)
    cell_circ_buf = cell_circ_buf.at[write_idx].set(avg_cell_circularity)
    cell_area_var_buf = cell_area_var_buf.at[write_idx].set(avg_cell_area_var)
    nuc_offset_buf = nuc_offset_buf.at[write_idx].set(avg_nucleus_offset)
    new_idx = (write_idx + 1) % window_size
    new_count = jnp.minimum(count + 1, window_size)

    updated_metric_history = (
        adv_buf, mi_buf, ent_buf, sense_buf, new_idx, new_count,
        intra_buf, adv_avg_buf, shape_buf, spread_buf,
        dark_range_buf, code_corr_buf, angle_spread_buf,
        nuc_ratio_buf, nuc_ecc_buf, cell_circ_buf, cell_area_var_buf,
        nuc_offset_buf,
    )
    updated_belief_space = belief_space[:6] + (updated_metric_history,)
    return updated_belief_space


@jax.jit
def compute_metric_slopes(belief_space):
    """Compute short/medium/long slopes from the metric history circular buffer.

    Returns a tuple of 13 slopes:
        (adv_short, mi_short, adv_med, mi_med, ent_long,
         sense_short, intra_short, adv_avg_short, sense_med,
         shape_short, shape_med, spread_short, spread_med)

    Short = last 20 generations, Medium = last 50, Long = full window (100).
    Uses least-squares linear regression.
    Returns 0.0 for any window that doesn't have enough data yet.
    """
    metric_history = belief_space[6]
    (adv_buf, mi_buf, ent_buf, sense_buf, write_idx, count,
     intra_buf, adv_avg_buf, shape_buf, spread_buf,
     dark_range_buf, code_corr_buf, angle_spread_buf,
     nuc_ratio_buf, nuc_ecc_buf, cell_circ_buf, cell_area_var_buf,
     nuc_offset_buf) = metric_history
    window_size = adv_buf.shape[0]

    # Recover gracefully if an older checkpoint already contains NaNs.
    adv_buf = jnp.nan_to_num(adv_buf, nan=0.0, posinf=0.0, neginf=0.0)
    mi_buf = jnp.nan_to_num(mi_buf, nan=0.0, posinf=0.0, neginf=0.0)
    ent_buf = jnp.nan_to_num(ent_buf, nan=0.0, posinf=0.0, neginf=0.0)
    sense_buf = jnp.nan_to_num(sense_buf, nan=0.0, posinf=0.0, neginf=0.0)
    intra_buf = jnp.nan_to_num(intra_buf, nan=0.0, posinf=0.0, neginf=0.0)
    adv_avg_buf = jnp.nan_to_num(adv_avg_buf, nan=0.0, posinf=0.0, neginf=0.0)
    shape_buf = jnp.nan_to_num(shape_buf, nan=0.0, posinf=0.0, neginf=0.0)
    spread_buf = jnp.nan_to_num(spread_buf, nan=0.0, posinf=0.0, neginf=0.0)
    _ = dark_range_buf
    _ = code_corr_buf
    _ = angle_spread_buf
    _ = nuc_ratio_buf
    _ = nuc_ecc_buf
    _ = cell_circ_buf
    _ = cell_area_var_buf
    _ = nuc_offset_buf

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
    shape_short   = _slope_over_last_n(shape_buf, 20, write_idx, count)
    shape_med     = _slope_over_last_n(shape_buf, 50, write_idx, count)
    spread_short  = _slope_over_last_n(spread_buf, 20, write_idx, count)
    spread_med    = _slope_over_last_n(spread_buf, 50, write_idx, count)

    return (adv_short, mi_short, adv_med, mi_med, ent_long,
            sense_short, intra_short, adv_avg_short, sense_med,
            shape_short, shape_med, spread_short, spread_med)


@jax.jit
def compute_semantic_trap_state(belief_space):
    """Estimate whether the run is separating codes via a semantic shortcut.

    A04a focuses on the known BloodMNIST failure mode: one shared prototype
    with code separation carried mainly by orientation.  The trap score is
    intentionally conservative and only activates once MI is at least mildly
    positive, so early random exploration is not treated as a shortcut.
    """
    metric_history = belief_space[6]
    (adv_buf, mi_buf, ent_buf, sense_buf, write_idx, count,
     intra_buf, adv_avg_buf, shape_buf, spread_buf,
     dark_range_buf, code_corr_buf, angle_spread_buf,
     nuc_ratio_buf, nuc_ecc_buf, cell_circ_buf, cell_area_var_buf,
     nuc_offset_buf) = metric_history
    window_size = adv_buf.shape[0]

    def _latest(buf):
        idx = (write_idx - 1) % window_size
        return jnp.where(count > 0, buf[idx], 0.0)

    def _slope(buf, n):
        has_enough = count >= n
        indices = (jnp.arange(n) + write_idx - n) % window_size
        vals = buf[indices]
        x = jnp.arange(n, dtype=jnp.float32)
        mean_x = jnp.mean(x)
        mean_y = jnp.mean(vals)
        numer = jnp.sum((x - mean_x) * (vals - mean_y))
        denom = jnp.sum((x - mean_x) ** 2) + 1e-12
        slope = numer / denom
        return jnp.where(has_enough, slope, 0.0)

    mi_latest = _latest(mi_buf)
    dark_latest = _latest(dark_range_buf)
    code_corr_latest = _latest(code_corr_buf)
    angle_spread_latest = _latest(angle_spread_buf)
    nuc_ratio_latest = _latest(nuc_ratio_buf)
    nuc_ecc_latest = _latest(nuc_ecc_buf)
    cell_circularity_latest = _latest(cell_circ_buf)
    cell_area_var_latest = _latest(cell_area_var_buf)
    nucleus_offset_latest = _latest(nuc_offset_buf)

    area_stability = jnp.clip(1.0 - (cell_area_var_latest / 0.02), 0.0, 1.0)
    bio_score_latest = (
        0.20 * nuc_ratio_latest
        + 0.15 * nuc_ecc_latest
        + 0.35 * cell_circularity_latest
        + 0.20 * nucleus_offset_latest
        + 0.10 * area_stability
    )

    dark_range_short = _slope(dark_range_buf, 20)
    code_corr_short = _slope(code_corr_buf, 20)
    angle_spread_short = _slope(angle_spread_buf, 20)

    # Heuristic trap model for the current BloodMNIST shortcut:
    #  - MI is present, so codes are carrying information
    #  - prototype correlation remains high
    #  - orientation spread is non-trivial
    #  - dark-range morphology diversity remains weak
    mi_ready = jnp.clip(mi_latest / 0.05, 0.0, 1.0)
    corr_high = jnp.clip((code_corr_latest - 0.62) / 0.12, 0.0, 1.0)
    angle_high = jnp.clip((angle_spread_latest - 0.18) / 0.22, 0.0, 1.0)
    dark_low = jnp.clip((0.05 - dark_latest) / 0.05, 0.0, 1.0)
    bio_low = jnp.clip((0.10 - bio_score_latest) / 0.10, 0.0, 1.0)
    circularity_low = jnp.clip((0.60 - cell_circularity_latest) / 0.25, 0.0, 1.0)
    area_var_high = jnp.clip((cell_area_var_latest - 0.004) / 0.012, 0.0, 1.0)
    offset_low = jnp.clip((0.08 - nucleus_offset_latest) / 0.08, 0.0, 1.0)

    trap_score = mi_ready * (
        0.28 * corr_high
        + 0.14 * angle_high
        + 0.08 * dark_low
        + 0.14 * bio_low
        + 0.20 * circularity_low
        + 0.08 * area_var_high
        + 0.08 * offset_low
    )
    trap_score = jnp.clip(trap_score, 0.0, 1.0)
    trap_active = trap_score > 0.45

    dom_penalty = 0.9 * trap_score
    hist_boost = 0.9 * trap_score
    topo_boost = 0.5 * trap_score
    stdev_boost = 1.0 + 0.20 * trap_score

    return (
        dark_range_short,
        code_corr_short,
        angle_spread_short,
        dark_latest,
        code_corr_latest,
        angle_spread_latest,
        cell_circularity_latest,
        cell_area_var_latest,
        nucleus_offset_latest,
        nuc_ratio_latest,
        nuc_ecc_latest,
        bio_score_latest,
        trap_score,
        trap_active.astype(jnp.float32),
        dom_penalty,
        hist_boost,
        topo_boost,
        stdev_boost,
    )


@jax.jit
def _stable_entropy_from_scores(class_scores: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy from class scores robustly.

    The input is treated as logits/scores. We convert to probabilities with
    softmax, then compute categorical entropy.
    """
    safe_scores = jnp.nan_to_num(class_scores, nan=0.0, posinf=0.0, neginf=0.0)
    probs = jax.nn.softmax(safe_scores)
    probs = jnp.clip(probs, 1e-8, 1.0)
    probs = probs / jnp.maximum(jnp.sum(probs), 1e-8)
    entropy = -jnp.sum(probs * jnp.log(probs))
    default_entropy = jnp.log(jnp.array(probs.shape[0], dtype=probs.dtype))
    return jnp.where(jnp.isfinite(entropy), entropy, default_entropy)


@jax.jit
def update_domain_ks(
    belief_space, best_solution, stdev, best_scaled_noise,
    best_fitness_adv, best_fitness_mi, best_fitness_combined,
    disc_logit, best_r_sense, best_r_cons, best_r_shape_div
):
    """Update Domain KS: Pareto archive with GAN diagnostic metadata.

    Maintains 20 non-dominated solutions ranked by
    [|adv|, |mi|, -shape_div, |entropy|].
    Each archived solution also tracks r_sense/r_cons/r_shape_div for
    failure-mode detection in adaptive guidance selection.
    """
    domain_ks = belief_space[1]
    (best_solutions, stdevs, best_scaled_noises,
     best_fitnesses_adv, best_fitnesses_mi, best_fitnesses_combined,
     entropies, r_senses, r_conses, r_shape_divs) = domain_ks

    # Append new solution
    updated_solutions = jnp.concatenate([best_solutions, best_solution], axis=0)
    updated_stdevs = jnp.concatenate([stdevs, stdev.reshape(1, -1)], axis=0)
    updated_noises = jnp.concatenate([best_scaled_noises, best_scaled_noise], axis=0)
    updated_adv = jnp.concatenate([best_fitnesses_adv, best_fitness_adv.flatten()], axis=0)
    updated_mi = jnp.concatenate([best_fitnesses_mi, best_fitness_mi.flatten()], axis=0)
    updated_combined = jnp.concatenate([best_fitnesses_combined, best_fitness_combined.flatten()], axis=0)

    entropy = jnp.array([_stable_entropy_from_scores(disc_logit)])
    updated_entropy = jnp.concatenate([entropies, entropy], axis=0)

    updated_r_sense = jnp.concatenate([r_senses, best_r_sense.flatten()], axis=0)
    updated_r_cons = jnp.concatenate([r_conses, best_r_cons.flatten()], axis=0)
    updated_r_shape_div = jnp.concatenate([r_shape_divs, best_r_shape_div.flatten()], axis=0)

    # Non-dominated sort on [|adv|, |mi|, -shape_div, |entropy|]
    # (minimization setting: negate shape_div so larger diversity is preferred).
    objectives = jnp.stack([
        jnp.abs(updated_adv),
        jnp.abs(updated_mi),
        -updated_r_shape_div,
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
        updated_r_shape_div[selected],
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

    entropy = jnp.array([_stable_entropy_from_scores(disc_logit)])

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
    belief_space,
    best_solution,
    stdev,
    scaled_noise,
    fitness_value_adv,
    fitness_value_mi,
    tchebyscheff_score,
    disc_logit,
    bio_score,
):
    history_ks = belief_space[3]

    (best_solutions, best_stdevs, best_scaled_noises,
     best_fitnesses_adversarial, best_fitnesses_mutual_info,
     best_fitnesses_tchebycheff, entropies, bio_scores) = history_ks

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

    entropy = jnp.array([_stable_entropy_from_scores(disc_logit)])

    #entropy = entropy.reshape(-1, 1)
    
    updated_entropy = jnp.concatenate(
        [entropies, entropy], axis=0
    )
    updated_bio_scores = jnp.concatenate(
        [bio_scores, bio_score.flatten()], axis=0
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
    selected_bio_scores = updated_bio_scores[selected_indices]

    updated_history_ks = (
        selected_best_solutions,
        selected_best_stdevs,
        selected_best_scaled_noises,
        selected_best_fitnesses_adversarial,
        selected_best_fitnesses_mutual_info,
        selected_best_fitnesses_tchebycheff,
        selected_entropy,
        selected_bio_scores,
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

def _quantile_1d(values, q):
    values = jnp.sort(jnp.ravel(jnp.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)))
    n = values.shape[0]
    idx = jnp.clip(jnp.floor(q * (n - 1)).astype(jnp.int32), 0, n - 1)
    return values[idx]


@jax.jit
def update_normative_ks(
    belief_space,
    fitness_scores,
    all_spreads,
    all_safety_ratios,
    cell_circularity,
    cell_area_var,
    nucleus_offset,
    t,
):
    normative_ks = belief_space[5]

    k = 32
    elite_indices = jnp.argsort(fitness_scores)[-k:]

    elite_ratios = all_safety_ratios[elite_indices]
    elite_ratios_clamped = jnp.minimum(elite_ratios, 3.0)
    min_ratios = jnp.min(elite_ratios_clamped, axis=(1, 2))
    current_elite_min_safety = jnp.mean(min_ratios)

    elite_spreads = all_spreads[elite_indices]
    current_elite_avg_spread = jnp.mean(elite_spreads)

    elite_cell_circularity = cell_circularity[elite_indices]
    elite_cell_area_var = cell_area_var[elite_indices]
    elite_nucleus_offset = nucleus_offset[elite_indices]

    old_spread, old_safety, old_cell_circularity_floor, old_cell_area_var_ceiling, old_nucleus_offset_low, old_nucleus_offset_high = normative_ks

    new_spread = 0.95 * old_spread + 0.05 * current_elite_avg_spread
    new_safety = 0.95 * old_safety + 0.05 * current_elite_min_safety
    new_safety = jnp.clip(new_safety, 1.1, 2.0)

    circ_floor_target = _quantile_1d(elite_cell_circularity, 0.25)
    area_ceiling_target = _quantile_1d(elite_cell_area_var, 0.75)
    offset_low_target = _quantile_1d(elite_nucleus_offset, 0.20)
    offset_high_target = _quantile_1d(elite_nucleus_offset, 0.80)

    warmup_done = jnp.asarray(t, dtype=jnp.int32) >= 20000
    tighten_rate = jnp.float32(0.10)
    relax_rate = jnp.where(warmup_done, jnp.float32(0.001), jnp.float32(0.10))

    eps_circ = jnp.float32(0.01)
    eps_area = jnp.float32(0.0002)
    eps_offset = jnp.float32(0.01)

    circ_update_rate = jnp.where(
        circ_floor_target > old_cell_circularity_floor + eps_circ,
        tighten_rate,
        relax_rate,
    )
    area_update_rate = jnp.where(
        area_ceiling_target < old_cell_area_var_ceiling - eps_area,
        tighten_rate,
        relax_rate,
    )
    offset_low_update_rate = jnp.where(
        offset_low_target > old_nucleus_offset_low + eps_offset,
        tighten_rate,
        relax_rate,
    )
    offset_high_update_rate = jnp.where(
        offset_high_target < old_nucleus_offset_high - eps_offset,
        tighten_rate,
        relax_rate,
    )

    new_cell_circularity_floor = (
        (1.0 - circ_update_rate) * old_cell_circularity_floor
        + circ_update_rate * circ_floor_target
    )
    new_cell_area_var_ceiling = (
        (1.0 - area_update_rate) * old_cell_area_var_ceiling
        + area_update_rate * area_ceiling_target
    )
    new_nucleus_offset_low = (
        (1.0 - offset_low_update_rate) * old_nucleus_offset_low
        + offset_low_update_rate * offset_low_target
    )
    new_nucleus_offset_high = (
        (1.0 - offset_high_update_rate) * old_nucleus_offset_high
        + offset_high_update_rate * offset_high_target
    )

    new_cell_circularity_floor = jnp.clip(new_cell_circularity_floor, 0.45, 0.80)
    new_cell_area_var_ceiling = jnp.clip(new_cell_area_var_ceiling, 0.0010, 0.0100)
    new_nucleus_offset_low = jnp.clip(new_nucleus_offset_low, 0.05, 0.25)
    new_nucleus_offset_high = jnp.clip(new_nucleus_offset_high, 0.15, 0.55)
    new_nucleus_offset_high = jnp.maximum(new_nucleus_offset_high, new_nucleus_offset_low + 0.05)

    updated_normative_ks = (
        new_spread,
        new_safety,
        new_cell_circularity_floor,
        new_cell_area_var_ceiling,
        new_nucleus_offset_low,
        new_nucleus_offset_high,
    )

    updated_belief_space_normative = (
        belief_space[:5] + (updated_normative_ks,) + belief_space[6:]
    )

    return updated_belief_space_normative


@jax.jit
def compute_normative_state(belief_space):
    metric_history = belief_space[6]
    normative_ks = belief_space[5]
    (adv_buf, mi_buf, ent_buf, sense_buf, write_idx, count,
     intra_buf, adv_avg_buf, shape_buf, spread_buf,
     dark_range_buf, code_corr_buf, angle_spread_buf,
     nuc_ratio_buf, nuc_ecc_buf, cell_circ_buf, cell_area_var_buf,
     nuc_offset_buf) = metric_history
    (_, _, cell_circularity_floor, cell_area_var_ceiling,
     nucleus_offset_low, nucleus_offset_high) = normative_ks
    window_size = adv_buf.shape[0]

    def _latest(buf):
        idx = (write_idx - 1) % window_size
        return jnp.where(count > 0, buf[idx], 0.0)

    cell_circularity_latest = _latest(cell_circ_buf)
    cell_area_var_latest = _latest(cell_area_var_buf)
    nucleus_offset_latest = _latest(nuc_offset_buf)

    circ_violation = jnp.clip(
        (cell_circularity_floor - cell_circularity_latest) / (cell_circularity_floor + 1e-6),
        0.0,
        1.0,
    )
    area_violation = jnp.clip(
        (cell_area_var_latest - cell_area_var_ceiling) / (cell_area_var_ceiling + 1e-6),
        0.0,
        1.0,
    )
    offset_below = jnp.clip(
        (nucleus_offset_low - nucleus_offset_latest) / (nucleus_offset_low + 1e-6),
        0.0,
        1.0,
    )
    offset_above = jnp.clip(
        (nucleus_offset_latest - nucleus_offset_high) / (nucleus_offset_high + 1e-6),
        0.0,
        1.0,
    )
    offset_violation = jnp.maximum(offset_below, offset_above)

    norm_violation = jnp.clip(
        0.45 * circ_violation + 0.25 * area_violation + 0.30 * offset_violation,
        0.0,
        1.0,
    )
    norm_active = norm_violation > 0.12

    dom_penalty = 0.55 * norm_violation
    hist_boost = 0.40 * norm_violation
    sit_boost = 0.25 * norm_violation
    stdev_scale = jnp.clip(1.0 - 0.15 * norm_violation, 0.85, 1.0)

    return (
        cell_circularity_floor,
        cell_area_var_ceiling,
        nucleus_offset_low,
        nucleus_offset_high,
        circ_violation,
        area_violation,
        offset_violation,
        norm_violation,
        norm_active.astype(jnp.float32),
        dom_penalty,
        hist_boost,
        sit_boost,
        stdev_scale,
    )

@jax.jit
def _domain_ks_select_index(domain_ks, entropy_long_slope, adv_med_slope):
    """Adaptive buffer: select which archived solution Domain KS suggests.

    The Domain KS embodies knowledge about GAN training dynamics.  Instead of
    always returning the first Pareto solution, it detects the current training
    regime and picks the most appropriate archived solution:

    - Mode collapse risk (entropy dropping): pick highest-entropy solution
      to recover diversity.
    - Stagnation (adv not improving): pick highest r_shape_div solution to
      recover within-code conditional diversity.
    - Normal progress: pick solution with best combined fitness (exploit
      the domain's governing rules).
    """
    entropies = domain_ks[6]     # (20,)
    r_shape_divs = domain_ks[9]  # (20,)
    combined = domain_ks[5]      # (20,)

    # Detect mode collapse: entropy slope is negative (dropping)
    collapse_risk = entropy_long_slope < -0.005

    # Detect stagnation: adversarial slope is near zero (no medium-term improvement)
    stagnation = jnp.abs(adv_med_slope) < 0.001

    # Select index based on detected regime
    # Priority: collapse > stagnation > normal
    # (collapse is the most dangerous failure mode)
    idx_entropy = jnp.argmax(entropies)      # highest entropy (recover diversity)
    idx_shape_div = jnp.argmax(r_shape_divs) # best conditional shape diversity
    idx_combined = jnp.argmax(combined)      # best overall (exploit)

    idx = jnp.where(collapse_risk, idx_entropy,
          jnp.where(stagnation, idx_shape_div, idx_combined))
    return idx


@jax.jit
def get_center_guidance(belief_space, t, center):
    domain_ks = belief_space[1]
    situational_ks = belief_space[2]
    history_ks = belief_space[3]
    topographic_ks = belief_space[4]

    # Compute slopes from metric history
    slopes = compute_metric_slopes(belief_space)
    (adv_short, mi_short, adv_med, mi_med, ent_long,
     _sense_short, _intra_short, _adv_avg_short, _sense_med,
     _shape_short, _shape_med, spread_short, spread_med) = slopes

    semantic_state = compute_semantic_trap_state(belief_space)
    (_, _, _, _, _, _, _, _, _, _, _, _,
     trap_score, trap_active, dom_penalty, hist_boost, topo_boost, _stdev_boost) = semantic_state
    normative_state = compute_normative_state(belief_space)
    (_, _, _, _, _, _, _, norm_violation, norm_active,
     norm_dom_penalty, norm_hist_boost, norm_sit_boost, _norm_stdev_scale) = normative_state

    # Score each KS based on current training dynamics
    sit_score = situational_score(adv_short, mi_short)
    hist_score = historical_score(ent_long, adv_short)
    topo_score = topographic_score(ent_long, adv_med, spread_short)
    dom_score = domain_score(adv_med, mi_med, ent_long, spread_med)

    # A04a rescue logic: if the run is separating codes via a likely shortcut,
    # reduce Domain confidence and bias guidance toward historical rescue and
    # topographic exploration.
    dom_score = jnp.maximum(dom_score - dom_penalty, 0.0)
    hist_score = hist_score + hist_boost
    topo_score = topo_score + topo_boost
    dom_score = jnp.maximum(dom_score - norm_dom_penalty, 0.0)
    hist_score = hist_score + norm_hist_boost
    sit_score = sit_score + norm_sit_boost

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

    # Historical KS: use biology-aware rescue under a detected semantic trap.
    history_entropy_idx = jnp.argmax(history_ks[6])
    history_bio_idx = jnp.argmax(history_ks[7])
    history_idx = jnp.where((trap_active + norm_active) > 0.5, history_bio_idx, history_entropy_idx)
    history_ks_center = history_ks[0][history_idx]

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

    # Compute slopes from metric history
    slopes = compute_metric_slopes(belief_space)
    (adv_short, mi_short, adv_med, mi_med, ent_long,
     _sense_short, _intra_short, _adv_avg_short, _sense_med,
     _shape_short, _shape_med, spread_short, spread_med) = slopes

    semantic_state = compute_semantic_trap_state(belief_space)
    (_, _, _, _, _, _, _, _, _, _, _, _,
     trap_score, trap_active, dom_penalty, hist_boost, topo_boost, stdev_boost) = semantic_state
    normative_state = compute_normative_state(belief_space)
    (_, _, _, _, _, _, _, norm_violation, norm_active,
     norm_dom_penalty, norm_hist_boost, norm_sit_boost, norm_stdev_scale) = normative_state

    sit_score = situational_score(adv_short, mi_short)
    hist_score = historical_score(ent_long, adv_short)
    topo_score = topographic_score(ent_long, adv_med, spread_short)
    dom_score = domain_score(adv_med, mi_med, ent_long, spread_med)

    dom_score = jnp.maximum(dom_score - dom_penalty, 0.0)
    hist_score = hist_score + hist_boost
    topo_score = topo_score + topo_boost
    dom_score = jnp.maximum(dom_score - norm_dom_penalty, 0.0)
    hist_score = hist_score + norm_hist_boost
    sit_score = sit_score + norm_sit_boost

    ks_scores = jnp.array([dom_score, sit_score, hist_score, topo_score])

    # CATGAME-inspired weighted distribution (see get_center_guidance)
    temperature = 2.0
    ks_weights = jax.nn.softmax(ks_scores / temperature)

    # Domain KS: adaptive buffer selection (same failure-mode logic)
    domain_idx = _domain_ks_select_index(domain_ks, ent_long, adv_med)
    domain_ks_stdev = domain_ks[1][domain_idx]

    situational_ks_stdev = situational_ks[1].flatten()

    history_entropy_idx = jnp.argmax(history_ks[6])
    history_bio_idx = jnp.argmax(history_ks[7])
    history_idx = jnp.where((trap_active + norm_active) > 0.5, history_bio_idx, history_entropy_idx)
    history_ks_stdev = history_ks[1][history_idx]

    # Topographic: conservative — keep current exploration rate
    topo_ks_stdev = stdev.flatten()

    guidance = (
        domain_ks_stdev * ks_weights[0]
        + situational_ks_stdev * ks_weights[1]
        + history_ks_stdev * ks_weights[2]
        + topo_ks_stdev * ks_weights[3]
    )
    guidance = guidance * stdev_boost * norm_stdev_scale

    max_index = jnp.argmax(ks_weights)
    return guidance, max_index
