
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
    def __init__(self, dx, num_moves=1):
        # super
        self.dx = dx

    def propose_move(self, r):
        if num_moves == 1:
            return 0

