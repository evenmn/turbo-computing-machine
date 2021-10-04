import numpy as np

class Sampler:
    """
    Sampler base class
    """
    def __init__(self, stillinger_lim=np.inf):
        self.stillinger_lim = stillinger_lim

    def set_forcefield(self, forcefield):
        self.forcefield = forcefield

    def propose_move(self, r, move):
        """
        Propose new move among the available types of moves
        """
        i = np.random.randint(len(r))  # which particle to move
        ai, ui = self.forcefield.eval_acc_energy_par(r, i)
        ri = r[i]
        r[i] += move.propose_move(ai)  # self.get_dr(ai)

        # Stillinger cluster criterion
        _, dd = self.forcefield.distance_vector_par(r, i)
        if np.min(dd) > self.stillinger_lim:  # remove particle from cluster
            r = np.delete(r, i, 0)
            self.da = np.zeros(3)
            self.du = 0
        else:
            ai_new, ui_new = self.forcefield.eval_acc_energy_par(r, i)
            self.da = ai_new - ai
            self.du = ui_new - ui
        return r

    def accept_move(self, move):
        """
        Decide if move should be accepted or
        rejected
        """
        p = self.get_acceptance_prob(move)
        if p > np.random.normal():
            return True
        return False


class Metropolis(Sampler):
    """
    Metropolis sampling, as proposed by Metropolis et al. (1953)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_acceptance_prob(self, move):
        return move.accept(self.da) * np.exp(self.du)


class Umbrella(Sampler):
    """
    Umbrella Sampling, like proposed by Torrie and Valleau (1977)

    L : int
        number of windows in each direction (L**3 windows)
    psi : func { ndarray, int }
        bias functions, Gaussian by default
        
    """
    def __init__(self, L, psi=None, **kwargs):
        super().__init__(**kwargs)

        self.L = L
        c_1d = np.linspace(0, 1, L)
        c_3d = np.meshgrid(c_1d, c_1d, c_1d)
        self.cs = np.vstack(map(np.ravel, c_3d))
        self.k = L - 1

        if psi is None:
            self.psi = self.gauss
        else:
            self.psi = psi

    def gauss(self, r, i):
        """
        Define bias functions (umbrellas) as Gaussians
        """
        sigma = (r - np.min(r)) / np.max(r)  # shift r to (0, 1)
        return np.exp(-self.k**2 * (sigma - self.cs[i])**2 / 2.)

    def get_normalization_constants(self, Fij):
        """
        Get normalization of bias functions using the overlap
        matrix. The overlap matrix has to be obtained during
        sampling.
        """
        return np.ones(self.L)

    def get_acceptance(self, move):
        return move.accept(self.da) * np.exp(self.du)

