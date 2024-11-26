from src.main import *


class BGK(LBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau = kwargs.get("tau")  # relaxation factor tau

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f, source=0, force=None):
        """
        BGK Collision
        """
        rho, u = self.macro_vars(f, force)
        f_eq = self.equilibrium(rho, u)
        # return (1 - 1 / self.tau) * f + (1 / self.tau) * f_eq + (1 - 1 / (2 * self.tau)) * source
        # return (1 - 1 / self.tau) * f + (1 / self.tau) * f_eq + source
        # return f - 1/self.tau*(f - f_eq) + source
        return f- 1/self.tau*(f-f_eq)+(1 - 1/2*self.tau)*source

class MRT(LBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tau1 = kwargs.get("tau1")
        self.tau2 = kwargs.get("tau2")

    def collision(self, f, source=0, force=None):
        pass