import numpy as np
from tqdm import trange

from .sampler import BruteForce
from .integrator import VelocityVerlet
from .forcefield import LennardJones


class MD:
    """
    Molecular Dynamics motion class
    """
    def __init__(self):
        pass

    def run(self, steps):
        # This function should take TmpName obj
        for self.t in trange(steps):
            r, v, a = self.integrator(r, v, a)
            # out
        return r, v, a

class MC:
    """
    Monte Carlo motion class
    """
    def __init__(self, forcefield=LennardJones(1, 1, 3)):
        self.forcefield = forcefield
        self.sampler = BruteForce()

    def set_sampling(self, sampling):
        self.sampler = sampler

    def run(self, steps, r, v, a, u):
        # This function should take TmpName obj
        for i in trange(steps):
            r_new = self.sampler.propose_move(r)
            u_new = self.forcefield.eval_energy(r_new)
            accept = self.sampler.accept_move(u, u_new)
            if accept:
                r = r_new
                u = u_new
                # out
        return r, v, a


