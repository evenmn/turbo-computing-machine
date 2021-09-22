import numpy as np
from tqdm import trange

from .sampler import BruteForce
from .integrator import VelocityVerlet
from .forcefield import LennardJones


class MD:
    """
    Molecular Dynamics motion class
    """
    def __init__(self, dt=0.01, forcefield=LennardJones(1, 1, 3)):
        self.forcefield = forcefield
        self.integrator = VelocityVerlet(dt, forcefield)

    def set_integrator(self, integrator):
        self.integrator = integrator

    def run(self, steps, r, v, a, u):
        for i in trange(steps):
            r, v, a = self.integrator(r, v, a)
            # out
        return r, v, a

class MC:
    """
    Monte Carlo motion class
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sampler = BruteForce()

    def set_sampling(self, sampling):
        self.sampler = sampler

    def run(steps, r, v, a, u):
        for i in range(steps):
            r_new = self.sampler.propose_move(r)
            u_new = self.forcefield.eval_energy(r_new)
            accept = self.sampler.accept_move(u, u_new)
            if accept:
                r = r_new
                u = u_new
                # out
        return r, v, a


