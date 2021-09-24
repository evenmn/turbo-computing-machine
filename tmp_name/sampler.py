import numpy as np

class Sampler:
    """
    Sampler base class
    """
    def __init__(self, dx=1.0, stillinger_lim=None):
        self.dx = dx
        self.stillinger_lim = stillinger_lim

    def set_forcefield(self, forcefield):
        self.forcefield = forcefield

    def propose_move(self, r):
        npar = len(r)
        ndim = len(r[0])
        i = np.random.randint(npar)
        ai, ui = self.forcefield.eval_acc_energy_par(r, i)
        ri = r[i]
        r[i] += self.get_dr(ai)

        # Stillinger cluster criterion
        if self.stillinger_lim is not None:
            _, dd = self.forcefield.distance_vector_par(r, i)
            if np.min(dd, axis=1) > self.stillinger_lim:
                r[i] = ri
                self.da = 0
                self.du = 0
        else:
            ai_new, ui_new = self.forcefield.eval_acc_energy_par(r, i)
            self.da = ai - ai_new
            self.du = ui - ui_new
        return r

    def accept_move(self):
        p = self.get_acceptance_prob()
        if p > np.random.normal():
            return True
        return False


class BruteForce(Sampler):
    """
    Brute-Force Monte Carlo sampling
    
    Only translation moves allowed
    """

    def get_dr(self, ai):
        """
        Get position change
        """
        return (np.random.random((3,)) - 0.5) * self.dx

    def get_acceptance_prob(self):
        return np.exp(self.du)


class ImportanceSampling(Sampler):
    """
    Metropolis-Hastings algorithm

    Only translation moves allowed
    """
    def __init__(self, Ddt=0.01, **kwargs):
        super().__init__(**kwargs)
        self.Ddt = 0.01

    def green_ratio(self):
        return np.exp(0.5 * self.da * self.eps) + 1

    def get_dr(self, ai):
        self.eps = self.Ddt * ai + np.random.normal((3,)) * self.dx
        return self.eps

    def get_acceptance_prob(self):
        p = np.exp(self.du)
        return = p * self.green_ratio()


class UmbrellaSampling(Sampler):
    """
    Umbrella sampling is a special case of the
    Metropolis-Hastings algorithm

    Translational moves only
    """
    def __init__(self, Ddt=0.01, **kwargs):
        super().__init__(**kwargs)
        self.Ddt = 0.01

    def get_dr(self, ai):
        pass

    def get_acceptance_prob(self):
        pass

class AVBMC(UmbrellaSampling):
    """
    Aggregate-volume biased Monte Carlo type of moves,
    sampling technique is Monte Carlo
    """
    def __init__(self, ptrans, pswap, **kwargs):
        super().__init__(**kwargs)
        assert ptrans + pswap - 1 < 0.01, \
               "Total probability has to be 1"

    def trans_move(self, ai):
        pass

    def swap_move(self, ai):
        pass

    def get_dr(self, ai):
        if np.random.random() < ptrans:
            return self.trans_move(ai)
        else:
            return self.swap_move(ai)

    def get_acceptance_prob(self):
        pass

class EBAVBMC(UmbrellaSampling):
    pass
