import random as jrandom
import jax.numpy as jnp
import jax

from typing import List, Tuple, Dict
from evojax.algo.cultural.knowledge_sources import (
    initialize_domain_ks,
    initialize_situational_ks,
    initialize_history_ks,
    initialize_topographic_ks,
    initialize_normative_ks,
    get_center_guidance,
    get_stdev_guidance,
)

def initialize_belief_space(
    param_size: int,
    population_size: int,
    key: int,
    num_iterations: int = 40,
    max_individuals: int = 6,
):
    belief_space = (
        jnp.array([population_size]),
        initialize_domain_ks(param_size),
        initialize_situational_ks(param_size),
        initialize_history_ks(param_size,num_iterations),
        initialize_topographic_ks(param_size, max_individuals),
        initialize_normative_ks(param_size)
    )
    return belief_space

@jax.jit
def get_updated_params(belief_space, center, stdev, t):
    combined_guidance_center = combine_center_guidance(
        belief_space, t, center
    )
    combined_guidance_stdev = combine_stdev_guidance(
        belief_space, t, stdev
    )

    new_center = combined_guidance_center
    new_stdev = combined_guidance_stdev

    return new_center, new_stdev


def combine_center_guidance(belief_space, t, center):
    return get_center_guidance(belief_space, t, center)


def combine_stdev_guidance(belief_space, t, stdev):
    return get_stdev_guidance(belief_space, t, stdev)
