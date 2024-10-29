import jax.numpy as jnp


def get_density(f_i):
    """
    Calculates local, macroscopic densities from discrete velocities
    Done by summing the discrete velocities at every grid point

    Args:
        f_i (jnp.array): array of discrete velocities

    Returns:
        jnp.array: array of local densities
    """
    return jnp.einsum('xyi->xy', f_i)


def get_velocity(f_i, force, c_i):
    """
    Calculates local, macroscopic velocity from discrete velocities

    Args:
        f_i (jnp.array): array of discrete velocities

    Returns:
        jnp.array: array of local macroscopic velocities
    """
    rho = jnp.einsum('xyi->xy', f_i)
    u = jnp.einsum('ai,xyi->axy', c_i, f_i) + 0.5 * force
    return u / rho
