from src.main import LBM

#    6   2   5
#      \ | /
#    3 - 0 - 1
#      / | \
#    7   4   8
##################################################################################
##Deprecated, define boundary conditions inside simulation class, per simulation##
##################################################################################

#need better object-oriented implementation of boundary conditions is necessary.
#in function, cannot refer to variables inside the object, like lattice velocities.
#currently at the mercy of whatever I provide.
def bounce_back_couette2D(f_i):
    # Bounce-back top wall
    u_max = 0.1
    f_i = f_i.at[:, -1, 7].set(f_i[:, -1, 5]-1/6*u_max)
    f_i = f_i.at[:, -1, 4].set(f_i[:, -1, 2])
    f_i = f_i.at[:, -1, 8].set(f_i[:, -1, 6]+1/6*u_max)
    # Bounce-back bottom wall
    f_i = f_i.at[:, 0, 6].set(f_i[:, 0, 8])
    f_i = f_i.at[:, 0, 2].set(f_i[:, 0, 4])
    f_i = f_i.at[:, 0, 5].set(f_i[:, 0, 7])
    return f_i

def bounce_back_tube2D(f_i):
    # Bounce-back top wall
    f_i = f_i.at[:, -1, 7].set(f_i[:, -1, 5])
    f_i = f_i.at[:, -1, 4].set(f_i[:, -1, 2])
    f_i = f_i.at[:, -1, 8].set(f_i[:, -1, 6])
    # Bounce-back bottom wall
    f_i = f_i.at[:, 0, 6].set(f_i[:, 0, 8])
    f_i = f_i.at[:, 0, 2].set(f_i[:, 0, 4])
    f_i = f_i.at[:, 0, 5].set(f_i[:, 0, 7])
    return f_i

class BoundaryConditions:
    def __init__(self, **kwargs):
        pass
def moving_wall(values, domain, data):
    pass

def poiseuille_inlet_rho(f_i, rho):
    pass

def poiseuille_outlet_rho(f_i, rho):
    pass