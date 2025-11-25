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
    key: jax.Array,
    num_iterations: int = 40,
    max_individuals: int = 6,
    features: int = 128,
    num_codes: int = 10,
):
    #key, subkey = jax.random.split(key)
    
    belief_space = (
        jnp.array([population_size]),
        initialize_domain_ks(param_size),
        initialize_situational_ks(param_size),
        initialize_history_ks(param_size,num_iterations),
        initialize_topographic_ks(features, key, num_codes),
        initialize_normative_ks(param_size)
    )
    return belief_space

@jax.jit
def get_updated_params(belief_space, center, stdev, t):
    combined_guidance_center = combine_center_guidance(
        belief_space, t, center
    )
    combined_guidance_stdev, min_index = combine_stdev_guidance(
        belief_space, t, stdev
    )

    new_center = combined_guidance_center
    new_stdev = combined_guidance_stdev

    return new_center, new_stdev, min_index


def combine_center_guidance(belief_space, t, center):
    return get_center_guidance(belief_space, t, center)


def combine_stdev_guidance(belief_space, t, stdev):
    return get_stdev_guidance(belief_space, t, stdev)
