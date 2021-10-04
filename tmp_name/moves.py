import numpy as np


class Moves:
    def __init__(self):
        pass

    def __call__(self, ai):
        pass


class Trans(Moves):
    """
    Brute-force translational move
    """

    def __init__(self, dx=0.01):
        self.dx = dx

    def propose_move(self, ai):
        return (np.random.random((3,)) - 0.5) * self.dx

    def accept(self, da):
        """
        Returns the transition probability ratio, e.i.,
        Tji/Tij
        """
        return 1.


class TransMH(Moves):
    """
    Translational move using the Metropolis-Hastings method
    """

    def __init__(self, dx=0.01, Ddt = 0.01):
        self.dx = dx
        self.Ddt = Ddt

    def propose_move(self, ai):
        self.eps = self.Ddt * ai + np.random.normal((3,)) * self.dx
        return self.eps

    def accept(self, da):
        """
        Ratio between new and old Green's function
        """
        return np.exp(0.5 * da.dot(self.eps)) + 1


class AVBMCIntraSwap(Moves):
    """
    Aggregate-volume biased Monte Carlo (AVBMC) intra swap
    move to be performed in canonical ensemble. See
    Chen and Siepmann.
    """

    def __init__(self, r_below, r_above):
        self.r_below = r_below
        self.r_above = r_above

    def propose_move(self, ai):
        return

class AVBMCInterSwap(Moves):
    """

    """
    pass

class EBAVBMCIntraSwap(Moves):
    """

    """
    pass

class EBAVBMCInterSwap(Moves):
    """

    """
    pass
