from src.utility_functions import *
from src.main import *


class BGK(LBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.u_max = kwargs.get('u_max', 0.04)
        self.tau = kwargs.get("tau")  # relaxation factor tau
        if self.tau <= 0.5:
            raise ValueError("tau must be greater than 0.5")
        # Derived constants / Transport coefficients
        self.kinematic_viscosity = self.lattice.cs2*(self.tau - 1/2)
        # self.dynamic_viscosity = self.kinematic_viscosity*self.rho0 #option 1 from macroscopic formula
        # self.dynamic_viscosity = self.rho0*self.lattice.cs2*(self.tau - 1/2) #option 2, eq. 4.17
        self.mach = self.u_max/self.lattice.cs
        if self.mach >= 1:
            raise ValueError("Resulting mach number is too high for a stable simulation")
        self.length_scale = kwargs.get("length_scale", min(self.dimensions))
        self.reynolds = self.length_scale * self.u_max/self.kinematic_viscosity
        self.phi_init = kwargs.get("phi_init")
        # Class initialisation message
        print("--Simulation class created--")
        print("Simulation name: ", self.sim_name)
        print("--Simulation parameters [lattice units] --")
        print("Mesh dimensions: ", self.dimensions)
        print("Kinematic Viscosity: ", self.kinematic_viscosity)
        print("--Dimensionless numbers--")
        print("Reynold's number: ", self.reynolds)
        print("Mach number: ", self.mach)


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
        time1 = time.time()

        f, g = self.initialize()
        for it in range(nt):
            f, g, f_prev, g_prev = self.update(f, g_prev=g, it=it)  # updates f
            if it % self.plot_every == 0 and self.write:
                self.write_disk(f, nt)
                self.write_disk(g, nt)
            if it % self.plot_every == 0 and it >= self.plot_from and self.draw_plots:
                self.plot(f, it, g=g)
        time2 = time.time()
        print(f"Completed in: {time2 - time1:.1f} s")
        self.post_loop(f, nt, g=g)
        return f

    @partial(jax.jit, static_argnums=0)
    def update(self, f_prev, **kwargs):
        """
        Updates distribution functions f and g.
        -Calculates the forcing term and source term.
        -Applies collision operator to f and g.
        -Streams f and g to neighbours
        -applies boundary conditions if defined.
        :param f_prev: distribution function f of previous iteration (*dim, q)
        :return: f_post_stream: distribution function f of current iteration (*dim, q), f_prev
        """
        it = kwargs.get("it")
        if self.debug:
            if it >= self.plot_from:
                print("Debug!")
        ## get g and forcing/sourcing terms
        g_prev = kwargs.get("g_prev")
        force_prev = self.force_term(f_prev, g=g_prev)
        source_prev = self.source_term(f_prev, force_prev)
        if self.debug:
            f_prev_debug = np.asarray(f_prev)
            rho_prev_debug = np.asarray(self.macro_vars(f_prev)[0])
            g_prev_debug = np.asarray(g_prev)
            phi_prev_debug = np.asarray(self.macro_vars(g_prev)[0])
            force_prev_debug = np.asarray(force_prev)
            source_prev_debug = np.asarray(source_prev)
        ## collision of f and g according to model
        f_post_col = self.collision(f_prev, source=source_prev, force=force_prev, g=g_prev)
        g_post_col = self.g_collision(g_prev, f=f_prev, force=force_prev)
        if self.debug:
            f_post_col_debug = np.asarray(f_post_col)
            rho_post_col_debug = np.asarray(self.macro_vars(f_post_col)[0])
            g_post_col_debug = np.asarray(g_post_col)
            phi_post_col_debug = np.asarray(self.macro_vars(g_post_col)[0])
        ## Optional pre-streaming boundary conditions
        f_post_col = self.apply_pre_bc(f_post_col, f_prev, g_pop=g_post_col)
        g_post_col = self.apply_pre_bc_g(g_post_col, g_prev, f_pop=f_post_col)
        # if self.debug:
        #     f_post_col_debug_prebc = np.asarray(f_post_col)
        #     rho_post_col_debug_prebc = np.asarray(self.macro_vars(f_post_col)[0])
        #     g_post_col_debug_prebc = np.asarray(g_post_col)
        #     phi_post_col_debug_prebc = np.asarray(self.macro_vars(g_post_col)[0])
        ## Streaming of f and g
        f_post_stream = self.stream(f_post_col)
        g_post_stream = self.stream(g_post_col)
        if self.debug:
            f_post_stream_debug = np.asarray(f_post_stream)
            rho_post_stream_debug = np.asarray(self.macro_vars(f_post_stream)[0])
            g_post_stream_debug = np.asarray(g_post_stream)
            phi_post_stream_debug = np.asarray(self.macro_vars(g_post_stream)[0])
        ## Apply boundary conditions
        f_post_stream = self.apply_bc(f_post_stream, f_post_col, g_pop=g_post_stream)
        g_post_stream = self.apply_bc_g(g_post_stream, g_post_col, f_pop=f_post_stream)
        if self.debug:
            f_post_stream_debug_bc = np.asarray(f_post_stream)
            rho_post_stream_debug_bc = np.asarray(self.macro_vars(f_post_stream)[0])
            g_post_stream_debug_bc = np.asarray(g_post_stream)
            phi_post_stream_debug_bc = np.asarray(self.macro_vars(g_post_stream)[0])
        return f_post_stream, g_post_stream, f_prev, g_prev

    def force_term(self, f, **kwargs):
        """
        Calculates the forcing term from the concentration gradient
        :param f:
        :param kwargs:
        :return:
        """
        g = kwargs.get("g")
        phi, _ = self.macro_vars(g)
        mu = self.chemical_potential(phi)
        force_term = mu[..., jnp.newaxis]*nabla(phi)
        return force_term

    def free_energy(self, phi):
        """
        Calculates Landau Free Energy. Van der Sman et al.
        :param phi:
        :return:
        """
        return self.param_A/2 * phi**2 + self.param_B/4 * phi**4 + self.kappa/2 * np.sum(nabla(phi)**2)

    def apply_bc_g(self, g, g_prev, **kwargs):
        """
        Apply boundary conditions to g population.
        if not specified in model class, will apply the same boundary conditions as f.
        :param g:
        :param g_prev:
        :param kwargs:
        :return:
        """
        return self.apply_bc(g, g_prev, **kwargs)

    def apply_pre_bc_g(self, g, g_prev, **kwargs):
        """
        Apply boundary conditions to g population before streaming step.
        If not specified in model class, will apply the same boundary conditions as f.
        :param g:
        :param g_prev:
        :param kwargs:
        :return:
        """
        return self.apply_pre_bc(g, g_prev, **kwargs)

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
        phi, u_phi = self.macro_vars(g, force)
        g_eq = self.g_equilibrium(phi, u)
        g_post_col = g - 1 / self.tau_phi * (g - g_eq)
        return g_post_col

    def plot(self, f, it, **kwargs):
        g = kwargs.get('g')
        rho, u = self.macro_vars(f)
        phi, _ = self.macro_vars(g)
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        # Plot velocity (magnitude or x-component of velocity vector)
        # plt.imshow(u[:,:,0].T, cmap='viridis')
        if self.collision_mask is not None:
            u_magnitude = jnp.where(self.collision_mask, 0, u_magnitude)
            u_magnitude = jnp.pad(u_magnitude, 2, "constant", constant_values=0)
        plt.imshow(u_magnitude.T, cmap='viridis')
        # plt.imshow(rho.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="Velocity magnitude")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.title("it:" + str(it) + " sum_rho:" + str(jnp.sum(rho)))
        plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg", dpi=250)
        plt.clf()
        # Plot order parameter phi
        plt.imshow(phi.T, cmap='viridis', vmin=-1, vmax=1)
        # plt.imshow(phi.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="Order Parameter")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.title("it:"+str(it)+" Order parameter phi"+" sum_phi:"+str(jnp.sum(phi)))
        plt.savefig(self.sav_dir+"/fig_2D_phi_it"+str(it)+".jpg", dpi=250)
        plt.clf()