import numpy as np

class Sampler:
    """
    Sampler base class
    """
    def __init__(self):
        # super
        pass

    def set_forcefield(self, forcefield):
        self.forcefield = forcefield


class BruteForce(Sampler):
    """
    Brute-Force Monte Carlo sampling
    
    Only translation moves allowed
    """
    def __init__(self, dx=1.0, num_moves=1):
        self.dx = dx
        self.num_moves = num_moves

    def propose_move(self, r, a):
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
    def __init__(self, dx=1.0, num_moves=1, stillinger_lim=3):
        self.dx = dx
        self.dtD = 0.01
        self.stillinger_lim = stillinger_lim
        self.num_moves = num_moves

    def green_ratio(self):
        return np.exp(0.5 * self.da * self.eps) + 1

    def propose_move(self, r, a):
        npar = len(r)
        ndim = len(r[0])
        if self.num_moves == 1:
            i = np.random.randint(npar)
            j = np.random.randint(ndim)
            self.eps = self.dtD * a[i, j] + np.random.normal() * self.dx
            r[i, j] += self.eps
            _, dd = self.forcefield.distance_matrix(r)
            if np.any(np.min(dd, axis=1) > self.stillinger_lim):
                r[i, j] -= self.eps
                self.da = 0
            else:
                a_new = self.forcefield.eval_acc(r)
                self.da = a[i, j] - a_new[i, j]
        return r

    def accept_move(self, u, u_new):
        p = np.exp(u_new - u)
        w = p * self.green_ratio()
        if w > np.random.normal():
            return True
        return False
