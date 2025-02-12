from src.utility_functions import *
# from datetime import timedelta
import matplotlib.colors as colors
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
                if self.any_nan(f):
                    raise ValueError("NaN encountered: simulation ended")
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

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
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

class PhaseField(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau_phi = kwargs.get("tau_phi")  # Relaxation factor for distribution g
        self.gamma = kwargs.get("gamma")  # Tunable parameter related to the interfacial tension
        if self.tau_phi <= 0.5:
            raise ValueError("tau_phi must be greater than 0.5")
        self.mobility_parameter = self.gamma * (self.tau_phi - 1 / 2)
        self.param_A = kwargs.get("param_A")
        self.param_B = kwargs.get("param_B")
        self.kappa = kwargs.get("kappa")
        self.density_fluids = kwargs.get("density_fluids", (1.0, 1.0))
        # self.viscosity_fluids = kwargs.get("viscosity_fluids", (density*self.kinematic_viscosity for density in self.density_fluids))
        self.viscosity_fluids = kwargs.get("viscosity_fluids", (0.05, 0.05)) #dynamic viscosity (not kinematic)

    def initialize(self):
        u = jnp.zeros(self.u_dimension)
        if self.u_innit is not None:
            u = self.u_innit
        phi = self.phi_init
        phi_1 = phi
        phi_2 = 1-phi_1
        rho = self.density_fluids[0] * phi_1 + self.density_fluids[1] * phi_2
        p = rho
        f = self.f_equilibrium(rho, u, p=p)
        g = self.g_equilibrium(phi, u)
        # initial force
        kin_visc = self.get_viscosity(phi)
        pressure_force = self.pressure_force(rho, p)
        viscous_force = self.viscous_force(rho, u, kin_visc)
        force_prev = pressure_force + viscous_force
        return f, g, force_prev

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

        f, g, force_prev = self.initialize()
        for it in range(nt):
            f, g, f_prev, g_prev, force_prev = self.update(f, g_prev=g, it=it, force_prev=force_prev)  # updates f
            if it % self.plot_every == 0 and self.write:
                self.write_disk(f, nt)
                self.write_disk(g, nt)
            if it % self.plot_every == 0 and it >= self.plot_from and self.draw_plots:
                self.plot(f, it, g=g, force=force_prev)
                if self.any_nan(f):
                    self.crash_script()
                    raise ValueError("NaN encountered: simulation ended")
        # display elapsed time
        time2 = time.time()
        elapsed_time = time2 - time1
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            print(
                f"Completed in: {hours} hour{"s" if hours > 1 else ""}, {minutes:02d} min, {seconds:02d} sec ({elapsed_time:.1f} s)")
        elif minutes > 0:
            print(f"Completed in: {minutes:02d} min, {seconds:02d} sec ({elapsed_time:.1f} s)")
        else:
            print(f"Completed in: {seconds:02d} sec")
        # run post-loop script
        self.post_loop(f, nt, g=g, force=force_prev)
        return f

    @partial(jax.jit, static_argnums=0)
    def update(self, f_prev, **kwargs):
        it = kwargs.get("it")
        if self.debug:
            if it >= self.plot_from:
                print("Debug!")
        ## get g, update forces and calculate source term
        g_prev = kwargs.get("g_prev")
        force_prev = kwargs.get("force_prev")
        force_next = self.force_term(f_prev, g=g_prev, force_prev=force_prev)
        source_prev = self.source_term(f_prev, g=g_prev, force=force_next)

        ## collision of f and g according to model
        f_post_col = self.collision(f_prev, source=source_prev, force=force_next, g=g_prev)
        g_post_col = self.g_collision(g_prev, f=f_prev, force=force_prev)

        ## Streaming of f and g
        f_post_stream = self.stream(f_post_col)
        g_post_stream = self.stream(g_post_col)

        ## Apply boundary conditions
        f_post_stream = self.apply_bc(f_post_stream, f_post_col, g_pop=g_post_stream, force=force_next)
        g_post_stream = self.apply_bc_g(g_post_stream, g_post_col, f_pop=f_post_stream, force=force_next)


        return f_post_stream, g_post_stream, f_prev, g_prev, force_next

    def macro_vars(self, f, force=None, **kwargs):
        if f is None:
            return None, None
        if force is None:
            raise(ValueError("Force must be provided in this model!"))
        phi = kwargs.get("phi")
        phi_1 = phi
        phi_2 = 1 - phi
        rho = self.density_fluids[0] * phi_1 + self.density_fluids[1] * phi_2
        u = jnp.dot(f, self.lattice.c.T) / rho[..., jnp.newaxis]
        u += force/(2*rho[..., jnp.newaxis])
        return rho, u

    def get_pressure(self, rho, f, **kwargs):
        pressure = rho*self.lattice.cs2*jnp.sum(f, axis=-1)
        return pressure

    def get_phi(self, g, **kwargs):
        phi = jnp.sum(g, axis=-1)
        return phi

    def get_viscosity(self, phi):
        phi_1 = phi
        phi_2 = 1-phi
        return self.viscosity_fluids[0]*phi_1 + self.viscosity_fluids[1]*phi_2

    def force_term(self, f, **kwargs):
        """
        calls the 4 force terms and adds them together.
        :param f_prev:
        :param kwargs:
        :return:
        """
        g = kwargs.get("g")
        force_prev = kwargs.get("force_prev")
        phi = self.get_phi(g)
        rho, u = self.macro_vars(f, force=force_prev, phi=phi)
        p = self.get_pressure(rho, f)
        kin_visc = self.get_viscosity(phi)
        total_force = 0
        total_force += self.pressure_force(rho, p)
        total_force += self.viscous_force(rho, u, kin_visc)
        # total_force += self.surface_tension_force(phi)
        return total_force

    def pressure_force(self, rho, p):
        pressure_force = (-p/rho)[..., jnp.newaxis] * nabla(rho)
        return pressure_force

    # def viscous_force(self, rho, u, kin_visc):
    #     viscous_force = kin_visc * (nabla_vector(u)+nabla_vector(u).T) * nabla(rho)
    #     return viscous_force

    def viscous_force(self, rho, u, kin_visc):
        nabla_u = nabla_vector(u)
        axes = list(range(nabla_u.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        nabla_u_T = jnp.transpose(nabla_u, axes=axes)
        viscous_term = kin_visc[..., jnp.newaxis, jnp.newaxis] * (nabla_u + nabla_u_T)
        # viscous_force = jnp.einsum('ijkl,ijl->ijk', viscous_term, nabla(rho))
        viscous_force = jnp.einsum('...kl,...l->...k', viscous_term, nabla(rho))
        return viscous_force

    def surface_tension_force(self, phi):
        # g = kwargs.get("g")
        # phi = get_phi(g)
        mu = - self.param_A * phi + self.param_B * phi ** 3 - self.kappa * laplacian(phi)
        # mu = 2*self.param_B * (1-phi)*(1-2*phi) - self.kappa * laplacian(phi)
        surface_tension_force = mu[..., jnp.newaxis] * nabla(phi)
        return surface_tension_force


    def body_force(self, f_prev, **kwargs):
        pass

    # def source_term(self, f_prev, **kwargs):
    #     force = kwargs.get("force")
    #     g = kwargs.get("g")
    #     phi = self.get_phi(g)
    #     rho, u = self.macro_vars(f_prev, force, phi=phi)
    #     # term1 = self.lattice.c.T - u / self.lattice.cs2
    #     term1 = self.lattice.c[..., jnp.newaxis, jnp.newaxis] - u[jnp.newaxis, ...] / self.lattice.cs2
    #     term2 = jnp.einsum("ji,...j->...i", self.lattice.c, u) / self.lattice.cs2
    #     term2 = term2 * self.lattice.c
    #     source_term = self.lattice.w*(term1 + term2)
    #     source_term = jnp.einsum("...ab,...a->...b", source_term, force/rho)
    #     return source_term
    def source_term(self, f, **kwargs):
        """
        Calculate the source term from the force density, eq. 6.5 krüger et al.
        :param f: lattice populations of shape (*dim, q)
        :param force: force term/density of shape (*dim, d)
        :return source_term of shape (*dim, q)
        """
        force = kwargs.get("force")
        g = kwargs.get("g")
        phi = self.get_phi(g)
        rho, u = self.macro_vars(f, force, phi=phi)
        cc = jnp.einsum("iq,jq->ijq", self.lattice.c, self.lattice.c)
        cc_diff = cc - (self.lattice.cs2 * jnp.eye(self.lattice.d)[...,jnp.newaxis])
        term1 = self.lattice.c/self.lattice.cs2
        term2 = jnp.einsum("abq,...b->...aq", cc_diff, u) / (self.lattice.cs2 ** 2)
        source_term = self.lattice.w*(term1 + term2)
        source_term = jnp.einsum("...ab,...a->...b", source_term, force)
        return source_term

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f, **kwargs):
        g = kwargs.get("g")
        phi = self.get_phi(g)
        source = kwargs.get("source")
        force = kwargs.get("force",)
        rho, u =self.macro_vars(f, force, phi=phi)
        p = self.get_pressure(rho, f)
        rho, u = self.macro_vars(f, force, phi=phi)
        f_eq = self.f_equilibrium(rho, u, p=p)
        return f - 1 / self.tau * (f - f_eq) + (1 - 1 / (2 * self.tau)) * source

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def g_collision(self, g, **kwargs):
        f = kwargs.get("f")
        force = kwargs.get("force")
        phi = self.get_phi(g)
        rho, u = self.macro_vars(f, force, phi=phi)
        g_eq = self.g_equilibrium(phi, u)
        g_post_col = g - 1 / self.tau_phi * (g - g_eq)
        return g_post_col

    def f_equilibrium(self, rho, u, **kwargs):
        # equation 83
        # checked
        p = kwargs.get("p")
        cu = jnp.einsum("ji,...j->...i", self.lattice.c, u)
        uu = jnp.einsum("...i,...i->...", u, u)
        term1 = (p / (rho*self.lattice.cs2))[..., jnp.newaxis]
        term2 = cu / self.lattice.cs2
        term3 = cu ** 2 / (2 * self.lattice.cs2 ** 2)
        term4 = (uu / (2 * self.lattice.cs2))[..., jnp.newaxis]
        f_eq = self.lattice.w * (term1 + term2 + term3 - term4)
        # f_eq = g_eq.at[..., 0].set(phi - jnp.sum(g_eq[..., 1:], axis=-1))
        return f_eq

    def g_equilibrium(self, phi, u, **kwargs):
        # equation 70
        # checked
        cu = jnp.einsum("ji,...j->...i", self.lattice.c, u)
        uu = jnp.einsum("...i,...i->...", u, u)
        term1 = 1
        term2 = cu / self.lattice.cs2
        term3 = cu**2 / (2*self.lattice.cs2**2)
        term4 = (uu / (2 * self.lattice.cs2))[..., jnp.newaxis]
        g_eq = self.lattice.w * phi[..., jnp.newaxis] * (term1 + term2 + term3 - term4)
        # g_eq = g_eq.at[..., 0].set(phi - jnp.sum(g_eq[..., 1:], axis=-1))
        return g_eq

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

    def post_loop(self, f, nt, **kwargs):
        self.plot(f, nt, **kwargs)
        if self.create_video:
            # velocity:
            images_to_gif(self.sav_dir, "fig_2D_U")
            # order parameter:
            images_to_gif(self.sav_dir, "fig_2D_phi")
        return f

    def plot(self, f, it, **kwargs):
        viridis = plt.cm.get_cmap("viridis")
        wall_color = [0.5, 0.5, 0.5, 1]
        colors_list = [wall_color]+[viridis(i) for i in range(viridis.N)]
        custom_cmap = colors.ListedColormap(colors_list)

        force=kwargs.get("force")
        g = kwargs.get('g')
        phi = self.get_phi(g)
        rho, u = self.macro_vars(f, force=force, phi=phi)
        pressure = self.get_pressure(rho, f)
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        ## plot slice of velocity profile
        plt.plot(self.y, u_magnitude[-3, :].T)
        plt.savefig(self.sav_dir + "/fig_1D_U_it" + str(it) + ".jpg", dpi=250)
        print(f"{max(u_magnitude[-3, :].T)}")
        plt.clf()
        ## Plot velocity (magnitude or x-component of velocity vector)
        # plt.imshow(u[:,:,0].T, cmap='viridis')
        if self.collision_mask is not None:
            u_magnitude = jnp.where(self.collision_mask, 0, u_magnitude)
            # u_magnitude = jnp.pad(u_magnitude, 2, "constant", constant_values=0)
            phi = jnp.where(self.collision_mask, -0.01, phi)
            # phi = jnp.pad(phi, 2, "constant", constant_values=0)
        plt.imshow(u_magnitude.T, cmap='viridis')
        # plt.imshow(rho.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="Velocity magnitude")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.title("it:" + str(it) + " sum_rho:" + str(jnp.sum(rho)))
        plt.savefig(self.sav_dir + "/fig_2D_U_it" + str(it) + ".jpg", dpi=250)
        plt.clf()
        ## Plot order parameter phi
        plt.imshow(phi.T, cmap=custom_cmap, vmin=-0.01, vmax=1)
        # plt.imshow(phi.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="Order Parameter")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.title("it:" + str(it) + " Order parameter phi" + " sum_phi:" + str(jnp.sum(phi)))
        plt.savefig(self.sav_dir + "/fig_2D_phi_it" + str(it) + ".jpg", dpi=250)
        plt.clf()
        ## plot pressure p
        # plt.imshow(pressure.T, cmap='viridis')
        plt.contourf(pressure.T, cmap='viridis', levels=40)
        # plt.imshow(rho.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="Velocity magnitude")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.title("it:" + str(it) + " sum_p:" + str(jnp.sum(pressure)))
        plt.savefig(self.sav_dir + "/fig_2D_P_it" + str(it) + ".jpg", dpi=250)
        plt.clf()
