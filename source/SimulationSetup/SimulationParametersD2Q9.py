import jax.numpy as jnp
import math


def h(theta):
    alpha = jnp.arccos((jnp.sin(theta)) ** 2)  # Alpha described on page 357
    omega = 2 * jnp.sign(math.pi / 2 - theta) * jnp.sqrt(jnp.cos(alpha / 3) * (1 - jnp.cos(alpha / 3)))
    return beta_tau * jnp.sqrt(2 * kappa * p_c) * omega


# Simulation scales
# Set scale
scale = 1
# Set plotting frequency
plot_every = 750 # 300 * scale
skip_it = 5000 # 20000 * scale
# Determine the size of the different regions
r_0: int = int(30 * scale)
r_1: int = int(50 * scale)
r_2: int = int(100 * scale)
# Determine the simulation domain, NX is the channel length and NY is the channel height
nx = r_0 + r_1 + r_2
ny = int(30 * scale)
nt = int(100000 * scale)
# Simulation parameters
kappa = 0.004
rho_c = 3.5
p_c = 0.125
beta_tau = 0.03
tau = 1
u_max = 0.1 / scale
Nu = (2 * tau - 1) / 6
rho0 = rho_c * (1 - jnp.sqrt(beta_tau))
Re = ny * u_max / Nu
# Parameters related to gravity
tilt_angle: float = 2 * math.pi * (60 / 360)
g_set = 0.000003
# Parameters related to surface tension
theta_1: float = 2 * math.pi * (150 / 360)  # contact angle we can set it in radians
theta_2: float = 2 * math.pi * (80 / 360)  # contact angle we can set it in radians
theta_3 = 2 * math.pi * (120 / 360)  # contact angle we can set it in radians
h1 = h(theta_1)
h2 = h(theta_2)
