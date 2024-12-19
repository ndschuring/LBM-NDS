from src.utility_functions import *
from src.main import *


class BGK(LBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = kwargs.get("tau")  # relaxation factor tau
        if self.tau <= 0.5:
            raise ValueError("tau must be greater than 0.5")
        # Derived constants / Transport coefficients
        self.kinematic_viscosity = self.lattice.cs2*(self.tau - 1/2)
        self.reynolds = None #TODO define Reynold's number from tau
        self.phi_init = kwargs.get("phi_init")


    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f, **kwargs):
        """
        BGK Collision
        """
        g=kwargs.get("g")
        source = kwargs.get("source", 0)
        force = kwargs.get("force", None)
        rho, u = self.macro_vars(f, force)
        if g is not None:
            phi, _ = self.macro_vars(g)
            f_eq = self.f_equilibrium(rho, u, phi=phi)
        else:
            f_eq = self.f_equilibrium(rho, u)
        # return (1 - 1 / self.tau) * f + (1 / self.tau) * f_eq + (1 - 1 / (2 * self.tau)) * source
        # return (1 - 1 / self.tau) * f + (1 / self.tau) * f_eq + source
        # return f - 1/self.tau*(f - f_eq) + source
        return f- 1/self.tau*(f-f_eq)+(1 - 1/(2*self.tau))*source

class BGKMulti(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau_phi = kwargs.get("tau_phi") # Relaxation factor for distribution g
        self.gamma = kwargs.get("gamma") # Tunable parameter related to the interfacial tension
        if self.tau_phi <= 0.5:
            raise ValueError("tau_phi must be greater than 0.5")
        self.mobility_parameter = self.gamma*(self.tau_phi - 1/2)
        self.param_A = kwargs.get("param_A")
        self.param_B = kwargs.get("param_B")
        self.kappa = kwargs.get("kappa")

    def initialize(self):
        """
        calculates initial state from equilibrium where initial macroscopic values are:
            1. Velocities = 0
            2. Densities = rho0
        For entire domain
        """
        u = jnp.zeros(self.u_dimension)
        rho = self.rho0 * jnp.ones(self.rho_dimension)
        phi = self.phi_init
        f = self.f_equilibrium(rho, u, phi=phi)
        g = self.g_equilibrium(phi, u)
        return f, g

    def run(self, nt):
        """
        Runs the model
        1. Initialise simulation
            Initialise()
        2. iterate over nt
            update()
        3. store or plots values
            write_disk()
            plot()
        """
        f, g = self.initialize()
        for it in range(nt):
            f, g, f_prev, g_prev = self.update(f, g_prev=g, it=it)  # updates f
            if it % self.plot_every == 0 and self.write:
                self.write_disk(f, nt)
                self.write_disk(g, nt)
            if it % self.plot_every == 0 and it >= self.plot_from:
                self.plot(f, it, g=g)
        return f

    @partial(jax.jit, static_argnums=0)
    def update(self, f_prev, **kwargs):
        """
        updates discrete velocities
            1. Calculate forcing term from g
                force_term()
            2. Calculate source term from force
                source_term()
            3. Collision step, apply force to f
                collision()
                g_collision()
            4. Apply pre-streaming boundary conditions (if applicable)
                apply_pre_bc()
            5. Stream discrete velocities to neighbours for both f and g
                stream(f)
                stream(g)
            6. Apply post-streaming boundary conditions to both f and g
                apply_bc(f)
                apply_bc(g)
        """
        it = kwargs.get("it")
        # if it >= 945:
        #     print("break")
        # get g and forcing/sourcing terms
        g_prev = kwargs.get("g_prev")
        force_prev = self.force_term(f_prev, g=g_prev)
        source_prev = self.source_term(f_prev, force_prev)
        # collision of f and g
        f_post_col = self.collision(f_prev, source=source_prev, force=force_prev, g=g_prev)
        g_post_col = self.g_collision(g_prev, f=f_prev, force=force_prev) #temp f_prev
        # Optional pre-streaming boundary conditions
        f_post_col = self.apply_pre_bc(f_post_col, f_prev)
        g_post_col = self.apply_pre_bc(g_post_col, g_prev)
        # Streaming of f and g
        f_post_stream = self.stream(f_post_col)
        g_post_stream = self.stream(g_post_col)
        # Apply boundary conditions
        f_post_stream = self.apply_bc(f_post_stream, f_post_col)
        g_post_stream = self.apply_bc(g_post_stream, g_post_col, g_tag=True)
        return f_post_stream, g_post_stream, f_prev, g_prev

    def force_term(self, f, **kwargs):
        g = kwargs.get("g")
        # rho, u = self.macro_vars(f)
        phi, _ = self.macro_vars(g)
        mu = self.chemical_potential(phi)
        force_term = mu[..., jnp.newaxis]*nabla(phi)
        return force_term

    def free_energy(self, phi): #TODO define free energy
        pass

    def chemical_potential(self, phi):
        """
        Calculates the chemical potential, mu.
        Mu is the derivative of the free energy over the order parameter phi.
        Mu = d_psi/d_phi
        :param phi: Order parameter, shape (*dim)
        :return: chemical potential, shape (*dim)
        """
        chemical_potential = - self.param_A*phi + self.param_B*phi**3 - self.kappa*laplacian(phi)
        return chemical_potential

    def f_equilibrium(self, rho, u, **kwargs):
        """
        Calculates the equilibrium distribution of f for 2 components
        According to equation 9.97 of the LBM book (Krüger et al.)
        :param rho: Density, shape: (*dim)
        :param u: Velocity, shape: (*dim, d)
        :param kwargs: optional arguments: order parameter phi
        :return: equilibrium distribution f_eq, shape: (*dim, q)
        """
        phi = kwargs.get("phi")
        mu = self.chemical_potential(phi)
        # definitive version of equation 3.54 of LBM book.
        wi_rho = jnp.einsum("i,...->...i", self.lattice.w, rho)
        cc = jnp.einsum("iq,jq->ijq", self.lattice.c, self.lattice.c)
        cc_diff = cc - (self.lattice.cs2 * jnp.eye(self.lattice.d)[:,:,jnp.newaxis])
        uc = jnp.einsum("...j,ji->...i", u, self.lattice.c)
        uu = jnp.einsum("...a,...b->...ab", u, u)
        term1 = 1
        term2 = ((phi*mu)/(rho*self.lattice.cs2))[..., jnp.newaxis]
        term3 = uc/self.lattice.cs2
        term4 = jnp.einsum("...ab,abq->...q",uu, cc_diff) / (2*self.lattice.cs2**2)
        f_eq = wi_rho * (term1 + term2 + term3 + term4)
        f_eq = f_eq.at[..., 0].set(rho - jnp.sum(f_eq[..., 1:], axis=-1))
        return f_eq

    def g_equilibrium(self, phi, u, **kwargs):
        """
        Calculates the equilibrium distribution of g from order parameter and velocity
        According to equation 9.99 from LBM book (Krüger et al.)
        :param phi: Order parameter, shape: (*dim)
        :param u: Velocity, shape: (*dim, d)
        :param kwargs: optional arguments: None
        :return: equilibrium distribution g_eq, shape: (*dim, q)
        """
        mu = self.chemical_potential(phi)
        # wi_phi = jnp.einsum("i,...->...i", self.lattice.w, phi)
        cc = jnp.einsum("iq,jq->ijq", self.lattice.c, self.lattice.c)
        cc_diff = cc - (self.lattice.cs2 * jnp.eye(self.lattice.d)[..., jnp.newaxis])
        cu = jnp.einsum("ji,...j->...i", self.lattice.c, u)
        uu = jnp.einsum("...a,...b->...ab", u, u)
        term1 = ((self.gamma*mu) / self.lattice.cs2)[..., jnp.newaxis]
        term2 = phi[..., jnp.newaxis]*cu / self.lattice.cs2
        term3 = phi[..., jnp.newaxis]*jnp.einsum("...ab,abq->...q", uu, cc_diff) / (2 * self.lattice.cs2 ** 2)
        g_eq = self.lattice.w * (term1 + term2 + term3)
        g_eq = g_eq.at[..., 0].set(phi - jnp.sum(g_eq[..., 1:], axis=-1))
        return g_eq

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def g_collision(self, g, **kwargs):
        f = kwargs.get("f")
        force = kwargs.get("force")
        rho, u = self.macro_vars(f, force)
        phi, _ = self.macro_vars(g)
        # Limiters to prevent phi from blowing up, don't use.
        # phi = jnp.where(phi > 1, 1, phi)
        # phi = jnp.where(phi < -1, -1, phi)
        g_eq = self.g_equilibrium(phi, u)
        g_post_col = g - 1 / self.tau_phi * (g - g_eq)
        return g_post_col

    # def macro_vars_g(self, g):
    #     phi = jnp.sum(g, axis=-1)
    #     # u = jnp.dot(g, self.lattice.c.T) / phi[..., jnp.newaxis] #velocity (divide by rho)
    #     u = jnp.dot(g, self.lattice.c.T)
    #     return phi, u

class MRT(LBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau1 = kwargs.get("tau1")
        self.tau2 = kwargs.get("tau2")

    def collision(self, f, source=0, force=None):
        pass