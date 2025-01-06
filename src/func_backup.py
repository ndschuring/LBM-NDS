# def source_term(self, f, force):
#     """
#     old source term calculation from @JeffreyBontes
#     """
#     rho, u = self.macro_vars(f, force)
#     ux, uy = u[:, :, 0], u[:, :, 1]
#     # ux, uy = u[0], u[1]
#     fx, fy = force[:, :, 0], force[:, :, 1]
#     cx, cy = self.lattice.c[0], self.lattice.c[1]
#     source_ = jnp.zeros((self.nx, self.ny, 9))
#     source_ = source_.at[:, :, 0].set(self.lattice.w[0] * (3 * (cx[0] * fx + cy[0] * fy) - 3 * (ux * fx + uy * fy) +
#                                                            9 * (cx[0] * cx[0] * ux * fx + cy[0] * cx[0] * uy * fx + cx[
#                 0] * cy[
#                                                                     0] * ux * fy + cy[0] * cy[0] * uy * fy)))
#     source_ = source_.at[:, :, 1].set(self.lattice.w[1] * (3 * (cx[1] * fx + cy[1] * fy) - 3 * (ux * fx + uy * fy) +
#                                                            9 * (cx[1] * cx[1] * ux * fx + cy[1] * cx[1] * uy * fx + cx[
#                 1] * cy[
#                                                                     1] * ux * fy + cy[1] * cy[1] * uy * fy)))
#     source_ = source_.at[:, :, 2].set(self.lattice.w[2] * (3 * (cx[2] * fx + cy[2] * fy) - 3 * (ux * fx + uy * fy) +
#                                                            9 * (cx[2] * cx[2] * ux * fx + cy[2] * cx[2] * uy * fx + cx[
#                 2] * cy[
#                                                                     2] * ux * fy + cy[2] * cy[2] * uy * fy)))
#     source_ = source_.at[:, :, 3].set(self.lattice.w[3] * (3 * (cx[3] * fx + cy[3] * fy) - 3 * (ux * fx + uy * fy) +
#                                                            9 * (cx[3] * cx[3] * ux * fx + cy[3] * cx[3] * uy * fx + cx[
#                 3] * cy[
#                                                                     3] * ux * fy + cy[3] * cy[3] * uy * fy)))
#     source_ = source_.at[:, :, 4].set(self.lattice.w[4] * (3 * (cx[4] * fx + cy[4] * fy) - 3 * (ux * fx + uy * fy) +
#                                                            9 * (cx[4] * cx[4] * ux * fx + cy[4] * cx[4] * uy * fx + cx[
#                 4] * cy[
#                                                                     4] * ux * fy + cy[4] * cy[4] * uy * fy)))
#     source_ = source_.at[:, :, 5].set(self.lattice.w[5] * (3 * (cx[5] * fx + cy[5] * fy) - 3 * (ux * fx + uy * fy) +
#                                                            9 * (cx[5] * cx[5] * ux * fx + cy[5] * cx[5] * uy * fx + cx[
#                 5] * cy[
#                                                                     5] * ux * fy + cy[5] * cy[5] * uy * fy)))
#     source_ = source_.at[:, :, 6].set(self.lattice.w[6] * (3 * (cx[6] * fx + cy[6] * fy) - 3 * (ux * fx + uy * fy) +
#                                                            9 * (cx[6] * cx[6] * ux * fx + cy[6] * cx[6] * uy * fx + cx[
#                 6] * cy[
#                                                                     6] * ux * fy + cy[6] * cy[6] * uy * fy)))
#     source_ = source_.at[:, :, 7].set(self.lattice.w[7] * (3 * (cx[7] * fx + cy[7] * fy) - 3 * (ux * fx + uy * fy) +
#                                                            9 * (cx[7] * cx[7] * ux * fx + cy[7] * cx[7] * uy * fx + cx[
#                 7] * cy[
#                                                                     7] * ux * fy + cy[7] * cy[7] * uy * fy)))
#     source_ = source_.at[:, :, 8].set(self.lattice.w[8] * (3 * (cx[8] * fx + cy[8] * fy) - 3 * (ux * fx + uy * fy) +
#                                                            9 * (cx[8] * cx[8] * ux * fx + cy[8] * cx[8] * uy * fx + cx[
#                 8] * cy[
#                                                                     8] * ux * fy + cy[8] * cy[8] * uy * fy)))
#     return source_
#
#  @partial(jax.jit, static_argnums=0)
#     def equilibrium_(self, rho, u):
#         # Scheme from LBM book, linear equilibrium with incompressible model from sample code
#         # Calculate the dot product of u and c
#         uc_dot = u[:, :, 0][:, :, jnp.newaxis] * self.lattice.c[0, :] + u[:, :, 1][:, :, jnp.newaxis] * self.lattice.c[1, :]
#         # Multiply by 3 and add rho
#         f_eq = self.lattice.w * (rho[:, :, jnp.newaxis] + 3 * uc_dot)
#         return f_eq
#
#     @partial(jax.jit, static_argnums=0)
#     def equilibrium_(self, rho, u):
#         # using equation 3.4 of LBM book
#         uc_dot = jnp.tensordot(u, self.lattice.c, axes=(-1, 0))
#         uu_dot = jnp.sum(jnp.square(u), axis=-1) / (2*self.lattice.cs2)
#         f_eq = self.lattice.w[jnp.newaxis, jnp.newaxis, :] * rho[:, :, jnp.newaxis] * (1 + (uc_dot/self.lattice.cs2) + ((uc_dot**2)/(2*self.lattice.cs2**2)) - uu_dot[:,:,jnp.newaxis])
#         return f_eq
#
#     @partial(jax.jit, static_argnums=0)
#     def equilibrium_(self, rho, u):
#         # using equation 3.54 of LBM book
#         uc_dot = jnp.tensordot(u, self.lattice.c, axes=(-1, 0))
#         cc_dot = jnp.transpose(self.lattice.c)[:, :, None] * jnp.transpose(self.lattice.c)[:, None, :]
#         cc_diff = cc_dot - (self.lattice.cs2 * jnp.eye(self.lattice.d))
#         uu = u[..., None] * u[..., None, :]
#         term1 = 1
#         term2 = uc_dot / self.lattice.cs2
#         term3 = jnp.tensordot(uu, cc_diff, axes=([[-1, -2], [-1, -2]])) / (2 * self.lattice.cs2 ** 2)
#         f_eq = self.lattice.w[jnp.newaxis, jnp.newaxis, :] * rho[:, :, jnp.newaxis] * (term1 + term2 + term3)
#         return f_eq
#
#     @partial(jax.jit, static_argnums=0)
#     def equilibrium_(self, rho, u):
#         # AI "optimized" version of above
#         # Pre-compute common terms
#         u_squared = jnp.sum(u ** 2, axis=-1)
#
#         # Compute cu using broadcasting
#         cu = jnp.einsum('id,xy...d->xyi', self.lattice.c.T, u)
#
#         # Compute equilibrium distribution
#         f_eq = self.lattice.w[None, None, :] * rho[..., None] * (
#                 1.0
#                 + cu / self.lattice.cs2
#                 + (cu ** 2 / (2 * self.lattice.cs2 ** 2))
#                 - u_squared[..., None] / (2 * self.lattice.cs2)
#         )
#         return f_eq
#
#     @partial(jax.jit, static_argnums=0)
#     def equilibrium_(self, rho, u):
#         # using equation 4.42 of LBM book (for incompressible flows)
#         # Only works for gravity driven poiseuille and couette
#         uc_dot = jnp.tensordot(u, self.lattice.c, axes=(-1, 0))
#         cc_dot = jnp.transpose(self.lattice.c)[:, :, None] * jnp.transpose(self.lattice.c)[:, None, :]
#         cc_diff = cc_dot - (self.lattice.cs2 * jnp.eye(self.lattice.d))
#         uu = u[..., None] * u[..., None, :]
#         term2 = uc_dot / self.lattice.cs2
#         term3 = jnp.tensordot(uu, cc_diff, axes=([[-1, -2], [-1, -2]])) / (2 * self.lattice.cs2 ** 2)
#         f_eq = self.lattice.w[jnp.newaxis, jnp.newaxis, :] * rho[:, :, jnp.newaxis] + self.lattice.w[jnp.newaxis, jnp.newaxis, :] * self.rho0_ones[:, :, jnp.newaxis] * (term2 + term3)
#         return f_eq
