import numpy as np

class Sampler:
    """
    Sampler base class
    """
    def __init__(self):
        # super
        pass


class BruteForce(Sampler):
    """
    Brute-Force Monte Carlo sampling
    
    Only translation moves allowed
    """
    def __init__(self, dx=100.0, num_moves=1):
        self.dx = dx
        self.num_moves = num_moves

    def propose_move(self, r):
        npar = len(r)
        ndim = len(r[0])
        if self.num_moves == 1:
            i = np.random.randint(npar)
            j = np.random.randint(ndim)
            eps = (np.random.random() - 0.5) * self.dx
            r[i, j] += eps
        return r

    def accept_move(self, u, u_new):
        p = np.exp(u_new - u)
        if p > np.random.normal():
            return True
        return False

class ImportanceSampling(Sampler):
    """
    Metropolis-Hastings algorithm

    Only translation moves allowed
    """
    def __init__(self, dx=100.0, num_moves=1):
        self.dx = dx
        self.num_moves = num_moves

    def propose_move(self, r):
        pass
