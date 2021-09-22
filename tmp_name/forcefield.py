import numpy as np


class LennardJones:
    """
    Lennard-Jones force field
    """
    def __init__(self, sigma=1, epsilon=1, cutoff=3):
        # super
        self.sigma = sigma
        self.epsilon = epsilon
        self.cutoff = cutoff
        self.cutoff2 = cutoff * cutoff
        self.cutoff_corr = cutoff**(-12) + cutoff**(-6)

    def distance_matrix(self, r):
        """
        Compute distance between all particles
        """
        npar = len(r)
        x, y = r[:, np.newaxis, :], r[np.newaxis, :, :]
        drAll = x - y                                 # distance vector matrix
        #drAll = self.boundary.checkDistance(drAll)  # check if satisfy bc
        distanceSqrdAll = np.einsum('ijk,ijk->ij', drAll, drAll)    # r^2

        # Pick the upper triangular elements only from the matrices and flatten
        upperTri = np.triu_indices(npar, 1)
        distanceSqrdHalf = distanceSqrdAll[upperTri]
        drHalf = drAll[upperTri]

        # Pick the components that are closer than the cutoff distance only
        indices = np.nonzero(distanceSqrdHalf<self.cutoff2)
        distanceSqrd = distanceSqrdHalf[indices]
        dr = drHalf[indices]
        return distanceSqrdAll, distanceSqrd, dr, indices

    def eval_energy(self, r):
        """
        Evaluate energy
        """
        distanceSqrdAll, distanceSqrd, dr, indices = self.distance_matrix(r)
        distancePowSixInv = np.nan_to_num(distanceSqrd**(-3))      # 1/r^6
        distancePowTwelveInv = distancePowSixInv**2                # 1/r^12
        return 4 * (distancePowTwelveInv - distancePowSixInv - self.cutoff_corr)

    def eval_acc(self, r):
        """
        Evaluate acceleration
        """
        npar = len(r)
        ndim = len(r[0])
        distanceSqrdAll, distanceSqrd, dr, indices = self.distance_matrix(r)
        distancePowSixInv = np.nan_to_num(distanceSqrd**(-3))      # 1/r^6
        distancePowTwelveInv = distancePowSixInv**2                # 1/r^12
        factor = np.divide(2 * distancePowTwelveInv - distancePowSixInv, distanceSqrd)            # (2/r^12 - 1/r^6)/r^2
        factor[factor == np.inf] = 0
        force = 24 * np.einsum('i,ij->ij', factor, dr)

        # 
        forceMatrix = np.zeros((npar, npar, ndim))
        upperTri = np.triu_indices(npar, 1)
        self.index = np.array(upperTri).T
        index = self.index[indices].T
        forceMatrix[(index[0], index[1])] = force
        forceMatrix[(index[1], index[0])] = -force

        return np.sum(forceMatrix, axis=1)
