from source.func.DivOperatorsWettingBoundary import laplacian, grad_sq
from source.SimulationSetup.SimulationParametersD2Q9 import *
from source.SimulationSetup.LatticeParametersD2Q9 import lattice_velocities, w_i, w_i_2, n_discrete_velocities


def p_b(rho):
    v_rho = (rho - rho_c) / rho_c
    return p_c * (v_rho + 1) * (v_rho + 1) * (3 * v_rho * v_rho - 2 * v_rho + 1 - 2 * beta_tau)


def equilibrium_(rho, u): #equilibrium for multi-phase
    cdot3u = 3 * jnp.einsum('ai,axy->xyi', lattice_velocities, u)
    usq = jnp.einsum('axy->xy', u * u)
    feq_1 = jnp.einsum('i,xy->xyi', w_i, rho) * (1 + cdot3u * (1 + .5 * cdot3u) - 1.5 * usq[..., jnp.newaxis])
    feq_2_ = p_b(rho) - 1 / 3 * rho - kappa * rho * laplacian(rho)
    feq_2 = 3 * jnp.einsum('i,xy->xyi', w_i, feq_2_)
    feq_3 = kappa * jnp.einsum('bi,bxy->xyi', w_i_2, grad_sq(rho))
    feq_new__ = feq_1 + feq_2 + feq_3
    feq_new_ = feq_new__.at[:, :, 0].set(0)
    feq_new = feq_new_.at[:, :, 0].set(rho - jnp.einsum('xyi->xy', feq_new_))
    return feq_new

def equilibrium(rho, u):
    """
    fᵢᵉ = ρ Wᵢ (1 + 3 cᵢ ⋅ u + 9/2 (cᵢ ⋅ u)² − 3/2 ||u||₂²)
    """
    cdot3u = 3 * jnp.einsum('ai,axy->xyi', lattice_velocities, u)
    usq = jnp.einsum('axy->xy', u * u)
    feq_new = jnp.einsum('i,xy->xyi', w_i, rho) * (1 + cdot3u * (1 + .5 * cdot3u) - 1.5 * usq[..., jnp.newaxis])
    return feq_new


def force_term(rho):
    force_g = -rho * g_set
    force_parr = -force_g * jnp.sin(tilt_angle)
    force_perp = force_g * jnp.cos(tilt_angle)
    force_components = jnp.stack((force_parr, force_perp), axis=0)
    return force_components


def source_term(u, force):
    source_1 = 3 * jnp.einsum('ai,axy->xyi', lattice_velocities, force)
    source_2__ = 3 * jnp.einsum('bi,bxy->xyi', lattice_velocities, u)
    source_2 = source_1 * source_2__
    source_3 = -3 * jnp.einsum('axy,axy->xy', u, force)
    source = (1 - 0.5 / tau) * (
            jnp.einsum('i,xyi->xyi', w_i, (source_1 + source_2)) + jnp.einsum('i,xy->xyi', w_i, source_3))
    return source


def collision_bgk(f_i, f_eq, source):
    return (1 - 1 / tau) * f_i + (1 / tau) * f_eq + source


def streaming(f_i):
    for i in range(n_discrete_velocities):
        f_i = f_i.at[:, :, i].set(
            jnp.roll(
                jnp.roll(f_i[:, :, i], lattice_velocities[0, i], axis=0, ),
                lattice_velocities[1, i], axis=1, ))
    return f_i
