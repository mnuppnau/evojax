# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import pickle
import jax
import numpy as np
from typing import Union
from typing import Tuple
from typing import Callable

import jax.numpy as jnp
from jax import tree_util
from flax.core import FrozenDict

def get_single_params_format_fn(init_params: FrozenDict) -> Callable:
    """Return a function that formats a single flat parameter vector into the model parameter tree."""
    flat, tree = tree_util.tree_flatten(init_params)
    shapes = [p.shape for p in flat]
    sizes = [np.prod(s) for s in shapes]
    cum_sizes = np.cumsum(sizes)
    
    def params_format_fn(params: jnp.ndarray) -> FrozenDict:
        # Split the 1D parameter vector at the right indices.
        splits = jnp.split(params, cum_sizes[:-1])
        # Reshape each split to match the corresponding original shape.
        params_reshaped = [split.reshape(shape) for split, shape in zip(splits, shapes)]
        return tree_util.tree_unflatten(tree, params_reshaped)
    
    return params_format_fn

def get_params_format_disc_fn(init_params: FrozenDict):
    flat, tree = tree_util.tree_flatten(init_params)

    sizes = np.asarray([p.size for p in flat], dtype=np.int64)  # <- always int, scalar size = 1
    params_sizes = np.cumsum(sizes, dtype=np.int64)
    split_idx = params_sizes.tolist()  # jnp.split likes python ints

    total = int(params_sizes[-1])

    def params_format_fn(params: jnp.ndarray) -> FrozenDict:
        parts = jnp.split(params, split_idx, axis=-1)[:-1]
        parts = [x.reshape(y.shape) for x, y in zip(parts, flat)]
        return tree_util.tree_unflatten(tree, parts)

    return total, params_format_fn


def get_params_format_fn(init_params: FrozenDict) -> Tuple[int, Callable]:
    """Return a function that formats the parameters into a correct format."""

    flat, tree = tree_util.tree_flatten(init_params)
    params_sizes = np.cumsum([np.prod(p.shape) for p in flat])

    def params_format_fn(params: jnp.ndarray) -> FrozenDict:
        params = tree_util.tree_map(
            lambda x, y: x.reshape(y.shape),
            jnp.split(params, params_sizes, axis=-1)[:-1],
            flat)
        return tree_util.tree_unflatten(tree, params)

    return params_sizes[-1], params_format_fn


def create_logger(name: str,
                  log_dir: str = None,
                  debug: bool = False) -> logging.Logger:
    """Create a logger.

    Args:
        name - Name of the logger.
        log_dir - The logger will also log to an external file in the specified
                  directory if specified.
        debug - If we should log in DEBUG mode.

    Returns:
        logging.RootLogger.
    """

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(name)s: %(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO, format=log_format)
    logger = logging.getLogger(name)
    if log_dir:
        log_file = os.path.join(log_dir, '{}.txt'.format(name))
        file_hdl = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt=log_format)
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    # Set level explicitly, otherwise the logger does not output.
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger


def load_model_gen(model_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load policy parameters from the specified directory.

    Args:
        model_dir - Directory to load the model from.
    Returns:
        A pair of parameters, the shapes of which are
        (param_size,) and (1 + 2 * batch_stats_size,).
    """

    model_file = os.path.join(model_dir, 'bp_model_gen.npz')
    if not os.path.exists(model_file):
        raise ValueError('Model file {} does not exist.')
    with np.load(model_file) as data:
        params = data['params']
        batch_stats = data['batch_stats']
    return params, batch_stats

def load_model_disc(model_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load policy parameters from the specified directory.

    Args:
        model_dir - Directory to load the model from.
    Returns:
        A pair of parameters, the shapes of which are
        (param_size,) and (1 + 2 * batch_stats_size,).
    """

    model_file = os.path.join(model_dir, 'bp_model_disc.npz')
    if not os.path.exists(model_file):
        raise ValueError('Model file {} does not exist.')
    with np.load(model_file) as data:
        params = data['params']
        batch_stats = data['batch_stats']
    return params, batch_stats

def save_model(model_dir: str,
               model_name: str,
               params: Union[np.ndarray, jnp.ndarray],
               obs_params: Union[np.ndarray, jnp.ndarray] = None,
               batch_stats: Union[np.ndarray, jnp.ndarray] = None,
               best: bool = False) -> None:
    """Save policy parameters to the specified directory.

    Args:
        model_dir - Directory to save the model.
        model_name - Filename.
        params - The parameters to save.
        obs_params - The observation parameters to save
        best - Whether to save a copy as best.npz.
    """

    model_file = os.path.join(model_dir, '{}.npz'.format(model_name))
    np.savez(model_file,
             params=np.array(params),
             obs_params=np.array(obs_params),
             batch_stats=np.array(batch_stats))
    if best:
        model_file = os.path.join(model_dir, 'best.npz')
        np.savez(model_file,
                 params=np.array(params),
                 obs_params=np.array(obs_params))


def save_lattices(log_dir: str,
                  file_name: str,
                  fitness_lattice: jnp.ndarray,
                  params_lattice: jnp.ndarray,
                  occupancy_lattice: jnp.ndarray) -> None:
    """Save QD method's lattices."""
    file_name = os.path.join(log_dir, '{}.npz'.format(file_name))
    np.savez(file_name,
             params_lattice=np.array(params_lattice),
             fitness_lattice=np.array(fitness_lattice),
             occupancy_lattice=np.array(occupancy_lattice))


def get_tensorboard_log_fn(
        log_dir: str) -> Callable[[int, jnp.ndarray, str], None]:
    """
    Returns a custom logging function for the `evojax` `Trainer`.
    The function logs rewards after every train/test iteration with Tensorboard.
    It tries to use `tensorflow` or `pytorch` as tensorboard provider.

    Args:
        log_dir - directory to save store the tensorboard logs
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        def log_with_pytorch(i: int, scores: jnp.ndarray, stage: str):
            with SummaryWriter(log_dir=log_dir) as writer:
                writer.add_scalar(
                    f"{stage}/score_min", scores.min().item(), global_step=i)
                writer.add_scalar(
                    f"{stage}/score_max", scores.max().item(), global_step=i)
                writer.add_scalar(
                    f"{stage}/score_mean", scores.mean().item(), global_step=i)
                writer.add_scalar(
                    f"{stage}/score_std", scores.std().item(), global_step=i)
                writer.add_histogram(
                    f"{stage}/score_distribution", np.array(scores),
                    global_step=i)

        return log_with_pytorch

    except ImportError:
        pass

    try:
        import tensorflow as tf

        def log_with_tf(i: int, scores: jnp.ndarray, stage: str):
            with tf.summary.SummaryWriter(log_dir=log_dir).as_default():
                tf.summary.scalar(
                    f"{stage}/score_min", scores.min().item(), step=i)
                tf.summary.scalar(
                    f"{stage}/score_max", scores.max().item(), step=i)
                tf.summary.scalar(
                    f"{stage}/score_mean", scores.mean().item(), step=i)
                tf.summary.scalar(
                    f"{stage}/score_std", scores.std().item(), step=i)
                tf.summary.histogram(
                    f"{stage}/score_distribution", np.array(scores), step=i)

        return log_with_tf

    except ImportError:
        pass

    raise ImportError(
        "Please install the tensorboard AND (tensorflow OR pytorch) "
        "packages to log the rewards to tensorboard")


def _to_numpy(pytree):
    """Recursively convert all JAX arrays in a pytree to numpy arrays."""
    return jax.tree_util.tree_map(
        lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
        pytree,
    )


def _to_jax(pytree):
    """Recursively convert all numpy arrays in a pytree back to JAX arrays."""
    return jax.tree_util.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
        pytree,
    )


def save_checkpoint(
    checkpoint_dir: str,
    iteration: int,
    solver_hn,
    params_disc,
    batch_stats_disc,
    opt_disc,
    prng_key: jnp.ndarray,
    logger: logging.Logger = None,
) -> str:
    """Save full training state to a pickle checkpoint.

    Everything is converted to numpy before pickling so the file
    is portable and does not depend on the JAX runtime.

    Args:
        checkpoint_dir: Directory to write the checkpoint into.
        iteration: Current training iteration (loop index).
        solver_hn: The PGPE_CA solver instance.
        params_disc: Discriminator parameter pytree (Flax).
        batch_stats_disc: Discriminator batch-stats pytree (Flax).
        opt_disc: Discriminator optax optimizer state.
        prng_key: Current PRNG key from the training loop.
        logger: Optional logger.

    Returns:
        The path of the written checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'iteration': int(iteration),
        # --- PGPE / HyperNet solver state ---
        'pgpe_center': np.array(solver_hn._center),
        'pgpe_stdev': np.array(solver_hn._stdev),
        'pgpe_t': int(solver_hn._t),
        'pgpe_opt_state': _to_numpy(solver_hn._opt_state),
        'belief_space': _to_numpy(solver_hn.belief_space),
        # --- Discriminator state ---
        'params_disc': _to_numpy(params_disc),
        'batch_stats_disc': _to_numpy(batch_stats_disc),
        'opt_disc': _to_numpy(opt_disc),
        # --- Misc ---
        'prng_key': np.array(prng_key),
    }

    path = os.path.join(checkpoint_dir, f'checkpoint_{iteration}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Also write a small symlink/pointer so we can find "latest" easily.
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pkl')
    # Atomic-ish overwrite: write tmp then rename.
    tmp_path = latest_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, latest_path)

    if logger:
        logger.info(f'Checkpoint saved at iteration {iteration} -> {path}')
    return path


def load_checkpoint(
    checkpoint_path: str,
    solver_hn,
    disc_params_ref,
    disc_batch_stats_ref,
    opt_disc_ref,
    logger: logging.Logger = None,
):
    """Restore full training state from a checkpoint.

    The ``*_ref`` arguments are "reference" pytrees used only to verify
    structural compatibility (they come from a fresh init).  The actual
    values are overwritten from the checkpoint.

    Args:
        checkpoint_path: Path to the .pkl checkpoint file.
        solver_hn: A freshly-created PGPE_CA solver (will be mutated).
        disc_params_ref: Freshly-inited Discriminator params (for structure).
        disc_batch_stats_ref: Freshly-inited batch stats (for structure).
        opt_disc_ref: Freshly-inited optax optimizer state (for structure).
        logger: Optional logger.

    Returns:
        (iteration, params_disc, batch_stats_disc, opt_disc, prng_key)
    """
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)

    # --- Restore PGPE solver state ---
    solver_hn._center = jnp.array(ckpt['pgpe_center'])
    solver_hn._stdev = jnp.array(ckpt['pgpe_stdev'])
    solver_hn._t = int(ckpt['pgpe_t'])
    solver_hn._opt_state = _to_jax(ckpt['pgpe_opt_state'])
    belief_space = _to_jax(ckpt['belief_space'])

    # Migrate older checkpoints: metric_history (element [6]), history_ks
    # (element [3]), and normative_ks (element [5]) may have fewer buffers
    # than the current code expects. Pad with sensible defaults to match the
    # current schema.
    EXPECTED_METRIC_HISTORY_LEN = 18
    metric_history = belief_space[6]
    if isinstance(metric_history, (tuple, list)) and len(metric_history) < EXPECTED_METRIC_HISTORY_LEN:
        old_len = len(metric_history)
        window_size = metric_history[0].shape[0]
        padding = tuple(jnp.zeros((window_size,)) for _ in range(EXPECTED_METRIC_HISTORY_LEN - old_len))
        metric_history = tuple(metric_history) + padding
        belief_space = belief_space[:6] + (metric_history,)
        if logger:
            logger.info('Migrated metric_history: padded %d -> %d elements', old_len, EXPECTED_METRIC_HISTORY_LEN)

    EXPECTED_HISTORY_KS_LEN = 8
    history_ks = belief_space[3]
    if isinstance(history_ks, (tuple, list)) and len(history_ks) < EXPECTED_HISTORY_KS_LEN:
        old_len = len(history_ks)
        history_len = history_ks[0].shape[0]
        padding = tuple(jnp.zeros((history_len,)) for _ in range(EXPECTED_HISTORY_KS_LEN - old_len))
        history_ks = tuple(history_ks) + padding
        belief_space = belief_space[:3] + (history_ks,) + belief_space[4:]
        if logger:
            logger.info('Migrated history_ks: padded %d -> %d elements', old_len, EXPECTED_HISTORY_KS_LEN)

    EXPECTED_NORMATIVE_KS_LEN = 6
    normative_ks = belief_space[5]
    if isinstance(normative_ks, (tuple, list)) and len(normative_ks) < EXPECTED_NORMATIVE_KS_LEN:
        old_len = len(normative_ks)
        padding = (
            jnp.float32(0.55),
            jnp.float32(0.0040),
            jnp.float32(0.10),
            jnp.float32(0.35),
        )
        normative_ks = tuple(normative_ks) + padding[:EXPECTED_NORMATIVE_KS_LEN - old_len]
        belief_space = belief_space[:5] + (normative_ks,) + belief_space[6:]
        if logger:
            logger.info('Migrated normative_ks: padded %d -> %d elements', old_len, EXPECTED_NORMATIVE_KS_LEN)

    solver_hn.belief_space = belief_space

    # --- Restore Discriminator state ---
    params_disc = _to_jax(ckpt['params_disc'])
    batch_stats_disc = _to_jax(ckpt['batch_stats_disc'])
    opt_disc = _to_jax(ckpt['opt_disc'])

    prng_key = jnp.array(ckpt['prng_key'])
    iteration = ckpt['iteration']

    if logger:
        logger.info(
            f'Checkpoint loaded from {checkpoint_path}, '
            f'resuming at iteration {iteration}')

    return iteration, params_disc, batch_stats_disc, opt_disc, prng_key
