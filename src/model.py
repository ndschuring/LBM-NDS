from src.main import *


class BGK(LBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = kwargs.get("tau")  # relaxation factor tau

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f, source=0):
        """
        BGK Collision
        """
        rho, u = self.macro_vars(f)
        f_eq = self.equilibrium(rho, u)
        return (1 - 1 / self.tau) * f + (1 / self.tau) * f_eq + (1 - 1 / (2 * self.tau)) * source

    # def collision(self, f):
    #     rho, u = self.macro_vars(f)
    #     f_eq = self.equilibrium(rho, u)
    #     f_neq = f - f_eq
    #     f_post_col = f - 1/self.tau * f_neq
    #     # if self.force is not None:
    #     #     f_post_col = self.apply_force(f_post_col, f_eq, rho, u)
    #     return f_post_col