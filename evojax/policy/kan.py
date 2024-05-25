import logging
from typing import List, Tuple
from flax import linen as nn
import jax.numpy as jnp
from jax import random, jit, vmap
from functools import partial

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn

@jax.jit
def solve_single_lstsq(A_single, B_single):
    AtA = jnp.dot(A_single.T, A_single)
    AtB = jnp.dot(A_single.T, B_single)
    single_solution = jax.scipy.linalg.solve(AtA, AtB, assume_a='pos')
    return single_solution

@jax.jit
def solve_full_lstsq(A_full, B_full):
    solve_full = jax.vmap(solve_single_lstsq, in_axes=(0, 0))
    full_solution = solve_full(A_full, B_full)
    return full_solution

@partial(jit, static_argnums=(2,))
def get_spline_basis(x_ext, grid, k=3):
    grid = jnp.expand_dims(grid, axis=2)
    x = jnp.expand_dims(x_ext, axis=1)

    basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
    
    for K in range(1, k+1):
        left_term = (x - grid[:, :-(K + 1)]) / (grid[:, K:-1] - grid[:, :-(K + 1)])
        right_term = (grid[:, K + 1:] - x) / (grid[:, K + 1:] - grid[:, 1:(-K)])
        
        basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]

    return basis_splines

class KANLayer(nn.Module):
    n_in: int = 2
    n_out: int = 5
    k: int = 3

    const_spl: float or bool = False
    const_res: float or bool = False
    residual: nn.Module = nn.swish
    
    noise_std: float = 0.1
    grid_e: float = 0.15

    def setup(self):
        init_G = 3
        init_knot = (-1, 1)
        h = (init_knot[1] - init_knot[0]) / init_G
        grid = jnp.arange(-self.k, init_G + self.k + 1, dtype=jnp.float32) * h + init_knot[0]
        grid = jnp.expand_dims(grid, axis=0)
        grid = jnp.tile(grid, (self.n_in*self.n_out, 1))
        self.grid = self.variable('state', 'grid', lambda: grid)
        self.c_basis = self.param('c_basis', nn.initializers.normal(stddev=self.noise_std), (self.n_in * self.n_out, self.grid.value.shape[1]-1-self.k))
        
        if isinstance(self.const_spl, float):
            self.c_spl = jnp.ones(self.n_in*self.n_out) * self.const_spl
        elif self.const_spl is False:
            self.c_spl = self.param('c_spl', nn.initializers.constant(1.0), (self.n_in * self.n_out,))

        if isinstance(self.const_res, float):
            self.c_res = jnp.ones(self.n_in * self.n_out) * self.const_res
        elif self.const_res is False:
            self.c_res = self.param('c_res', nn.initializers.constant(1.0), (self.n_in * self.n_out,))

    def basis(self, x):
        batch = x.shape[0]
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
        x_ext = jnp.transpose(x_ext, (1, 0))
        grid = self.grid.value
        k = self.k
        bases = get_spline_basis(x_ext, grid, k)
        return bases

    def new_coeffs(self, x, ciBi):
        A = self.basis(x)
        Bj = jnp.transpose(A, (0, 2, 1))
        ciBi = jnp.expand_dims(ciBi, axis=-1)
        cj = solve_full_lstsq(Bj, ciBi)
        cj = jnp.squeeze(cj, axis=-1)
        return cj

    def update_grid(self, x, G_new):
        Bi = self.basis(x)
        ci = self.c_basis
        ciBi = jnp.einsum('ij,ijk->ik', ci, Bi)
        batch = x.shape[0]
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
        x_ext = jnp.transpose(x_ext, (1, 0))
        x_sorted = jnp.sort(x_ext, axis=1)
        ids = jnp.concatenate((jnp.floor(batch / G_new * jnp.arange(G_new)).astype(int), jnp.array([-1])))
        grid_adaptive = x_sorted[:, ids]
        margin = 0.01
        uniform_step = (x_sorted[:, -1] - x_sorted[:, 0] + 2 * margin) / G_new
        grid_uniform = (jnp.arange(G_new + 1, dtype=jnp.float32) * uniform_step[:, None] + x_sorted[:, 0][:, None] - margin)
        grid = self.grid_e * grid_uniform + (1.0 - self.grid_e) * grid_adaptive
        h = (grid[:, [-1]] - grid[:, [0]]) / G_new
        left = jnp.squeeze((jnp.arange(self.k, 0, -1)*h[:,None]), axis=1)
        right = jnp.squeeze((jnp.arange(1, self.k+1)*h[:,None]), axis=1)
        grid = jnp.concatenate([grid[:, [0]] - left, grid, grid[:, [-1]] + right], axis=1)
        self.grid.value = grid
        cj = self.new_coeffs(x, ciBi)
        return cj

    def __call__(self, x):
        batch = x.shape[0]
        x_ext = jnp.einsum('ij,k->ikj', x, jnp.ones(self.n_out,)).reshape((batch, self.n_in * self.n_out))
        x_ext = jnp.transpose(x_ext, (1, 0))
        res = jnp.transpose(self.residual(x_ext), (1, 0))
        Bi = self.basis(x)
        ci = self.c_basis
        spl = jnp.einsum('ij,ijk->ik', ci, Bi)
        spl = jnp.transpose(spl, (1, 0))
        cnst_spl = jnp.expand_dims(self.c_spl, axis=0)
        cnst_res = jnp.expand_dims(self.c_res, axis=0)
        y = (cnst_spl * spl) + (cnst_res * res)
        y_reshaped = jnp.reshape(y, (batch, self.n_out, self.n_in))
        y = (1.0 / self.n_in) * jnp.sum(y_reshaped, axis=2)
        grid_reshaped = self.grid.value.reshape(self.n_out, self.n_in, -1)
        input_norm = grid_reshaped[:, :, -1] - grid_reshaped[:, :, 0] + 1e-5
        spl_reshaped = jnp.reshape(spl, (batch, self.n_out, self.n_in))
        spl_reg = (jnp.mean(jnp.abs(spl_reshaped), axis=0)) / input_norm
        return y, spl_reg

class KAN(nn.Module):
    layer_dims: List[int]
    add_bias: bool = True
    
    k: int = 3
    const_spl: float or bool = False
    const_res: float or bool = False
    residual: nn.Module = nn.swish
    noise_std: float = 0.1
    grid_e: float = 0.15

    @nn.compact
    def __call__(self, x):
        spl_regs = []
        for i in range(len(self.layer_dims) - 1):
            layer = KANLayer(
                n_in=self.layer_dims[i],
                n_out=self.layer_dims[i + 1],
                k=self.k,
                const_spl=self.const_spl,
                const_res=self.const_res,
                residual=self.residual,
                noise_std=self.noise_std,
                grid_e=self.grid_e,
            )
            x, spl_reg = layer(x)
            if self.add_bias:
                bias = self.param(f'bias_{i}', nn.initializers.zeros, (self.layer_dims[i + 1],))
                x += bias
            spl_regs.append(spl_reg)
        return x, spl_regs

class KANPolicy(PolicyNetwork):
    def __init__(self, layer_dims, k=3, const_spl=False, const_res=False, residual=nn.swish, noise_std=0.1, grid_e=0.15, logger=None):
        if logger is None:
            self._logger = create_logger('KANPolicy')
        else:
            self._logger = logger

        self.model = KAN(
            layer_dims=layer_dims,
            k=k,
            const_spl=const_spl,
            const_res=const_res,
            residual=residual,
            noise_std=noise_std,
            grid_e=grid_e,
        )
        params = self.model.init(random.PRNGKey(0), jnp.zeros([1, layer_dims[0]]))
        self.num_params, format_params_fn = get_params_format_fn(params)
        self._logger.info(f'KANPolicy.num_params = {self.num_params}')
        self._format_params_fn = vmap(format_params_fn)
        self._forward_fn = vmap(self.model.apply, in_axes=(None, 0))

    def get_actions(self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        logits, activations = self._forward_fn({'params': params}, t_states.obs)
        return logits, p_states, activations
