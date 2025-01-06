import re
import jax
import jax.numpy as jnp

class Lattice:
    def __init__(self, name):
        self.name = name
        self.d, self.q = map(int, re.findall(r"\d+", name))
        # Construct the properties of the lattice
        self.c = jnp.array(self.lattice_velocity())
        self.w = jnp.array(self.lattice_weight())
        self.opp_indices = jnp.array(self.opposite_indices())
        self.main_indices = jnp.array(self.main_indices())
        self.right_indices = jnp.array(self.right_indices())
        self.left_indices = jnp.array(self.left_indices())
        if self.d >= 2:
            self.top_indices = jnp.array(self.top_indices())
            self.bottom_indices = jnp.array(self.bottom_indices())
        if self.d >= 3:
            self.front_indices = jnp.array(self.front_indices())
            self.back_indices = jnp.array(self.back_indices())

    def lattice_velocity(self):
        if self.name == "D1Q3":
            ci = jnp.array([0, 1, -1]) # Velocities x-component
        elif self.name == "D2Q9":
            ci = jnp.array([[0, 1, 0,-1, 0, 1,-1,-1, 1],  # Velocities x-components
                            [0, 0, 1, 0,-1, 1, 1,-1,-1]])  # Velocities y-components
        elif self.name == "D3Q27":
            ci = jnp.array([
                [0, 1, -1, 0,  0, 0,  0, 1, -1, 1, -1, 0,  0,  1, -1,  1, -1,  0,  0, 1, -1,  1, -1,  1, -1, -1,  1], #Velocities x-components
                [0, 0,  0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1,  0,  0,  1, -1, 1, -1,  1, -1, -1,  1,  1, -1], #Velocities y-components
                [0, 0,  0, 0,  0, 1, -1, 0,  0, 1, -1, 1, -1,  0,  0, -1,  1, -1,  1, 1, -1, -1,  1,  1, -1,  1, -1]  #Velocities z-components
            ])
        else:
            raise ValueError("This Lattice type is not supported")
        return ci

    def lattice_weight(self):
        if self.name == "D1Q3":
            wi = jnp.array([2 / 3, 1 / 3, 1 / 3])
        elif self.name == "D2Q9":
            wi = jnp.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
        elif self.name == "D3Q27":
            wi = jnp.array([8 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27,
                           1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54,
                           1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216])
        else:
            raise ValueError("This Lattice type is not supported")
        return wi

    def opposite_indices(self):
        c = self.c.T
        opposite = jnp.array([jnp.where(jnp.all(c == -c[i], axis=-1))[0][0] for i in range(self.q)])
        return opposite

    def main_indices(self):
        c = self.c.T
        if self.d == 2:
            return jnp.where(jnp.abs(c[:, 0]) + jnp.abs(c[:, 1]) == 1)[0]
        elif self.d == 3:
            return jnp.where(jnp.abs(c[:, 0]) + jnp.abs(c[:, 1]) + jnp.abs(c[:, 2]) == 1)[0]

    def right_indices(self):
        c = self.c.T
        return jnp.where(c[:, 0] == 1)[0]

    def left_indices(self):
        c = self.c.T
        return jnp.where(c[:, 0] == -1)[0]

    def top_indices(self):
        c = self.c.T
        return jnp.where(c[:, 1] == 1)[0]

    def bottom_indices(self):
        c = self.c.T
        return jnp.where(c[:, 1] == -1)[0]

    def front_indices(self):
        c = self.c.T
        return jnp.where(c[:, 2] == 1)[0]

    def back_indices(self):
        c = self.c.T
        return jnp.where(c[:, 2] == -1)[0]


class LatticeD2Q9(Lattice):
    def __init__(self):
        super().__init__("D2Q9")
        self._set_constants()

    def _set_constants(self):
        self.cs = 1 / jnp.sqrt(3)
        self.cs2 = 1 / 3
        self.inv_cs2 = 3
        self.i = jnp.arange(self.q)

class LatticeD3Q27(Lattice):
    def __init__(self):
        super().__init__("D3Q27")
        self._set_constants()

    def _set_constants(self):
        self.cs = 1 / jnp.sqrt(3)
        self.cs2 = 1 / 3
        self.inv_cs2 = 3
        self.i = jnp.arange(self.q)

