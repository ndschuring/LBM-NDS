from src.utility_functions import *
from functools import partial
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import datetime
import time
import jax
import sys
import os


# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")
# os.environ['XLA_FLAGS'] = "--xla_disable_hlo_passes=constant_folding"


class LBM:
    def __init__(self, **kwargs):
        # Initialisation Parameters
        self.nx = kwargs.get("nx") #x dimension
        self.ny = kwargs.get("ny") #y dimension
        self.nz = kwargs.get("nz") #z dimension
        self.lattice = kwargs.get("lattice") #set lattice
        # if an obstacle mask is provided, reset dimensions
        self.collision_mask = kwargs.get('collision_mask', None)
        if self.collision_mask is not None:
            self.bounce_mask = self.get_bounce_mask(self.collision_mask)
            if self.lattice.d == 2:
                self.nx, self.ny, _ = self.bounce_mask.shape
            elif self.lattice.d == 3:
                self.nx, self.ny, self.nz, _ = self.bounce_mask.shape
        # set dimensions based on lattice
        self.dimensions = [self.nx or 1, self.ny or 1, self.nz or 1]
        self.dimensions = self.dimensions[:self.lattice.d]
        self.rho_dimension = tuple(self.dimensions)
        self.u_dimension = tuple((*self.dimensions, self.lattice.d))
        # density for initialisation
        self.u_innit = kwargs.get('u_innit')
        self.rho0 = kwargs.get("rho0", 1) # scalar rho0
        self.rho0_ones = jnp.ones(self.rho_dimension)*self.rho0 # matrix rho0
        # Gravity Parameters
        self.g_set = kwargs.get("g_set", 0) # gravitational constant
        self.tilt_angle = kwargs.get("tilt_angle", 0) # angle of system if gravity
        # Plotting Parameters
        self.x = jnp.arange(1, self.nx+1) - 0.5
        self.y = jnp.arange(1, self.ny+1) - 0.5
        self.z = jnp.arange(1, self.nz+1) - 0.5
        self.plot_every = kwargs.get("plot_every", 50)
        self.plot_from = kwargs.get("plot_from", 0)
        self.sim_name = str(self)
        # Path location for storing results
        today = datetime.datetime.now().strftime(f"{self.sim_name}-%Y-%m-%d_%H-%M-%S")
        cwd = os.path.abspath(__file__)
        self.sav_dir = os.path.join(os.path.dirname(cwd), "../test", today)
        if not os.path.isdir(self.sav_dir):
            os.makedirs(self.sav_dir)
        # Boolean parameters
        self.draw_plots = kwargs.get("draw_plots", True)
        self.create_video = kwargs.get("create_video", True)
        self.write = kwargs.get("write", False)
        self.debug  = sys.monitoring.get_tool(sys.monitoring.DEBUGGER_ID) is not None
        if self.debug:
            jax.config.update("jax_disable_jit", True)
        self.multiphase_state = kwargs.get("multiphase_state", False)

    def any_nan(self, arr):
        return jnp.any(jnp.isnan(arr))

    def __str__(self):
        """
        Fallback name if unspecified in simulation class
        :param: None
        :return: String representation of self
        """
        return "undefined_sim"

    def get_bounce_mask(self, collision_mask):
        if self.lattice.d == 2:
            shifted_mask = jnp.stack([
                jnp.roll(collision_mask, shift=(dx, dy), axis=(0, 1))
                for dx, dy in self.lattice.c.T
            ], axis=-1)
            boundary_mask = (~shifted_mask[:, :, 0]) & jnp.any(shifted_mask[:, :, 1:], axis=-1)
            bounce_mask = boundary_mask[..., jnp.newaxis] & shifted_mask
            return bounce_mask
        elif self.lattice.d == 3:
            shifted_mask = jnp.stack([
                jnp.roll(collision_mask, shift=(dx, dy, dz), axis=(0, 1, 2))
                for dx, dy, dz in self.lattice.c.T
            ], axis=-1)
            boundary_mask = (~shifted_mask[:, :, :, 0]) & jnp.any(shifted_mask[:, :, :, 1:], axis=-1)
            bounce_mask = boundary_mask[..., jnp.newaxis] & shifted_mask
            return bounce_mask

    def run(self, nt):
        """
        Runs the model, iterating for nt iterations.
        Plots and writes to disk if specified.
        :param nt: number of iterations
        :return: final distribution function of shape (*dim, q)
        """
        # Initialise f of simulation
        time1 = time.time()
        f = self.initialize()
        for it in range(nt):
            f, f_prev = self.update(f) #updates f
            if it % self.plot_every == 0 and self.write:
                self.write_disk(f, nt)
            if it % self.plot_every == 0 and it >= self.plot_from and self.draw_plots:
                self.plot(f, it)
                if self.any_nan(f):
                    self.crash_script()
                    raise ValueError("NaN encountered: simulation ended")
        time2 = time.time()
        print(f"Completed in: {time2 - time1:.1f} s")
        self.post_loop(f, nt)
        return f

    def crash_script(self):
        parent_dir = os.path.dirname(self.sav_dir)
        current_folder = os.path.basename(self.sav_dir)
        new_folder = f"{current_folder} - crashed"
        new_save_dir = os.path.join(parent_dir, new_folder)
        os.rename(self.sav_dir, new_save_dir)

    def initialize(self):
        """
        calculates initial state from equilibrium where the initial macroscopic values are:
            1. Velocities = 0
            2. Densities = rho0
        For entire domain
        :param: None
        :return: initial distribution function of shape (*dim, q)
        """
        u = jnp.zeros(self.u_dimension)
        if self.u_innit is not None:
            u = self.u_innit
        rho = self.rho0 * jnp.ones(self.rho_dimension)
        f = self.f_equilibrium(rho, u)
        return f

    @partial(jax.jit, static_argnums=0)
    def update(self, f_prev):
        """
        Updates discrete velocities of distribution function f.
        -Calculates the forcing term and source term.
        -Applies collision operator.
        -Streams to neighbour
        -applies boundary conditions if defined.
        :param f_prev: distribution function f of previous iteration (*dim, q)
        :return: f_post_stream: distribution function f of current iteration (*dim, q), f_prev
        """
        # Calculate forcing/sourcing terms
        force_prev = self.force_term(f_prev)
        source_prev = self.source_term(f_prev, force_prev)
        if self.debug:
            f_prev_debug = np.asarray(f_prev)
            rho_prev_debug = np.asarray(self.macro_vars(f_prev)[0])
            force_prev_debug = np.asarray(force_prev)
            source_prev_debug = np.asarray(source_prev)
        # Collision of f according to model
        f_post_col = self.collision(f_prev, source=source_prev, force=force_prev)
        if self.debug:
            f_post_col_debug = np.asarray(f_post_col)
            rho_post_col_debug = np.asarray(self.macro_vars(f_post_col)[0])
        # Optional pre-streaming boundary conditions
        f_post_col = self.apply_pre_bc(f_post_col, f_prev)
        # if self.debug:
            # f_post_col_debug_pre = np.asarray(f_post_col)
            # rho_post_col_debug_pre = np.asarray(self.macro_vars(f_post_col)[0])
        # Streaming of f
        f_post_stream = self.stream(f_post_col)
        if self.debug:
            f_post_stream_debug = np.asarray(f_post_stream)
            rho_post_stream_debug = np.asarray(self.macro_vars(f_post_stream)[0])
        # Apply boundary conditions
        f_post_stream = self.apply_bc(f_post_stream, f_post_col)
        if self.debug:
            f_post_stream_debug_post = np.asarray(f_post_stream)
            rho_post_stream_debug_post = np.asarray(self.macro_vars(f_post_stream)[0])
        return f_post_stream, f_prev

    @partial(jax.jit, static_argnums=0, inline=True)
    def macro_vars(self, f, force=None):
        """
        Calculate macroscopic variables of f using method of moments
        0th moment: density rho
        1st moment: momentum
        Moments of g
        0th moment: phi
        :param f: distribution function f of shape (*dim, q)
        :param force: force density of shape (*dim, d)
        :return: macroscopic variables of f, rho of shape (*dim), u of shape (*dim, d)
        """
        if f == None:
            return None, None
        rho = jnp.sum(f, axis=-1)
        u = jnp.dot(f, self.lattice.c.T) / rho[..., jnp.newaxis] #velocity (divide by rho)
        if force is not None:
            u += force/(2*rho[..., jnp.newaxis])
        return rho, u

    def collision(self, f,  **kwargs):
        """
        --Specified in model class--
        Applies collision operator
        """
        return f

    def apply_bc(self, f, f_prev, **kwargs):
        """
        --Specified in simulation class--
        Applies boundary conditions after streaming
        if not defined, returns post-streaming populations
        """
        return f

    def force_term(self, f, **kwargs):
        """
        Force term for gravity force.
        :param f: distribution function f of shape (*dim, q)
        :param kwargs: None
        :return: Components of force term of shape (*dim, d)
        """
        rho, u = self.macro_vars(f)
        # Gravity Force xy
        force_g = - rho * self.g_set
        force_parr = - force_g * jnp.sin(self.tilt_angle)
        force_perp = force_g * jnp.cos(self.tilt_angle)
        if self.lattice.d == 3:
            force_redundant = jnp.zeros_like(force_parr)
            force_components = jnp.stack((force_parr, force_perp, force_redundant), axis=-1)
        else:
            force_components = jnp.stack((force_parr, force_perp), axis=-1)
        return force_components

    def source_term(self, f, force):
        """
        Calculate the source term from the force density, eq. 6.5 krüger et al.
        :param f: lattice populations of shape (*dim, q)
        :param force: force term/density of shape (*dim, d)
        :return source_term of shape (*dim, q)
        """
        rho, u = self.macro_vars(f, force)
        cc = jnp.einsum("iq,jq->ijq", self.lattice.c, self.lattice.c)
        cc_diff = cc - (self.lattice.cs2 * jnp.eye(self.lattice.d)[...,jnp.newaxis])
        term1 = self.lattice.c/self.lattice.cs2
        term2 = jnp.einsum("abq,...b->...aq", cc_diff, u) / (self.lattice.cs2 ** 2)
        source_term = self.lattice.w*(term1 + term2)
        source_term = jnp.einsum("...ab,...a->...b", source_term, force)
        return source_term

    def apply_pre_bc(self, f, f_prev, **kwargs):
        """
        --Specified in simulation class--
        Applies boundary conditions before streaming
        if not defined, returns post-collision populations
        """
        return f

    @partial(jax.jit, static_argnums=(0,))
    def stream(self, f):
        """
        Streams discrete velocities to neighbours in up to 3D
        Applies inherent periodic BC from rolling matrix
        """
        def stream_i(f_i, c):
            if self.lattice.d == 1:
                return jnp.roll(f_i, (c[0]), axis=0)
            if self.lattice.d == 2:
                return jnp.roll(f_i, (c[0], c[1]), axis=(0, 1))
            if self.lattice.d == 3:
                return jnp.roll(f_i, (c[0], c[1], c[2]), axis=(0, 1, 2))
        return jax.vmap(stream_i, in_axes=(-1, 0), out_axes=-1)(f, self.lattice.c.T)

    def f_equilibrium(self, rho, u, **kwargs):
        """
        Calculates the equilibrium distribution of f for a single phase
        According to equation 3.54 of the LBM book (Krüger et al.)
        :param rho: Density, shape: (*dim)
        :param u: Velocity, shape: (*dim, d)
        :param kwargs: optional arguments: None
        :return: equilibrium distribution f_eq of shape: (*dim, q)
        """
        # definitive version of equation 3.54 of LBM book. Utilizing jnp.einsum to actually understand what is going on.
        wi_rho = jnp.einsum("i,...->...i", self.lattice.w, rho)
        cc = jnp.einsum("iq,jq->ijq", self.lattice.c, self.lattice.c)
        cc_diff = cc - (self.lattice.cs2 * jnp.eye(self.lattice.d)[...,jnp.newaxis])
        uc = jnp.einsum("...j,ji->...i", u, self.lattice.c)
        uu = jnp.einsum("...a,...b->...ab", u, u)
        term1 = 1
        term2 = uc/self.lattice.cs2
        term3 = jnp.einsum("...ab,abq->...q",uu, cc_diff) / (2*self.lattice.cs2**2)
        f_eq = wi_rho * (term1 + term2 + term3)
        # f_eq = f_eq.at[..., 0].set(rho - jnp.sum(f_eq[..., 1:], axis=-1)) #correction term to ensure mass conservation?
        return f_eq

    def g_equilibrium(self, phi, u, **kwargs):
        """
        --Specified in model class--
        Calculates the equilibrium distribution of g
        """
        pass

    def post_loop(self, f, nt, **kwargs):
        self.plot(f, nt, **kwargs)
        if self.create_video:
            # velocity:
            images_to_gif(self.sav_dir, "fig_2D")
        return f

    def write_disk(self, f, nt, **kwargs):
        """
        store macroscopic values in array for use of outside visualisation software.
        Writing an XML of some kind for VTK to be implemented in ParaView
        Unfinished
        """
        rho, u = self.macro_vars(f)
        # def save_vtk_timestep(u, rho, nt, output_path):
        #     # Create VTK grid
        #     grid = vtk.vtkstructuredGrid()
        #     grid.SetDimensions(self.nx, self.ny, 1)

            # create points
            # # Add velocity data
            # vel_flat = np.zeros((velocity.shape[0] * velocity.shape[1], 3))
            # vel_flat[:, 0] = velocity[:, :, 0].flatten()
            # vel_flat[:, 1] = velocity[:, :, 1].flatten()
            # vtk_velocity = numpy_support.numpy_to_vtk(vel_flat)
            # vtk_velocity.SetName("velocity")
            #
            # # Add density data
            # vtk_density = numpy_support.numpy_to_vtk(density.flatten())
            # vtk_density.SetName("density")
            #
            # # Attach data to grid
            # grid.GetPointData().AddArray(vtk_velocity)
            # grid.GetPointData().AddArray(vtk_density)
            #
            # # Write to file
            # writer = vtk.vtkXMLImageDataWriter()
            # writer.SetFileName(f"{output_path}/lbm_t{timestep:04d}.vti")
            # writer.SetInputData(grid)
            # writer.Write()
        pass

    def plot(self, f, it, **kwargs):
        """
        Default plotter function, 
        Specify bespoke plotter functions in simulation class
        :param f: lattice populations of shape (*dim, q)
        :param it: iteration number
        :param kwargs: Optional arguments: None
        :return: None
        """
        rho, u = self.macro_vars(f)
        u_magnitude = jnp.linalg.norm(u, axis=-1, ord=2)
        # Plot velocity (magnitude or x-component of velocity vector)
        # plt.imshow(u[:,:,0].T, cmap='viridis')
        if self.collision_mask is not None:
            u_magnitude = jnp.where(self.collision_mask, 0, u_magnitude)
            u_magnitude = jnp.pad(u_magnitude, 2, "constant", constant_values=0)
        plt.imshow(u_magnitude.T, cmap='viridis')
        # plt.imshow(rho.T, cmap='viridis')
        plt.gca().invert_yaxis()
        plt.colorbar(label="velocity magnitude")
        plt.xlabel("x [lattice units]")
        plt.ylabel("y [lattice units]")
        plt.title("it:" + str(it) + "sum_rho:" + str(jnp.sum(rho)))
        plt.savefig(self.sav_dir + "/fig_2D_it" + str(it) + ".jpg", dpi=250)
        plt.clf()