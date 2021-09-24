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
        Compute distance between all particles squared
        """
        x, y = r[:, np.newaxis, :], r[np.newaxis, :, :]
        dr = x - y                             # distance vector matrix
        #dr = self.boundary.checkDistance(dr)  # check if satisfy bc
        distanceSqrd = np.einsum('ijk,ijk->ij', dr, dr)    # r^2
        return dr, distanceSqrd

    def distance_matrix_triu(self, r):
        """
        Compute distance between all particles
        """
        npar = len(r)
        drAll, distanceSqrdAll = self.distance_matrix(r)

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
        Evaluate energy of entire system
        """
        distanceSqrdAll, distanceSqrd, dr, indices = self.distance_matrix_triu(r)
        distancePowSixInv = np.nan_to_num(distanceSqrd**(-3))      # 1/r^6
        distancePowTwelveInv = distancePowSixInv**2                # 1/r^12
        return np.sum(4 * (distancePowTwelveInv - distancePowSixInv - self.cutoff_corr))

    def eval_acc(self, r):
        """
        Evaluate acceleration of all particles
        """
        npar = len(r)
        ndim = len(r[0])
        distanceSqrdAll, distanceSqrd, dr, indices = self.distance_matrix_triu(r)
        distancePowSixInv = np.nan_to_num(distanceSqrd**(-3))      # 1/r^6
        distancePowTwelveInv = distancePowSixInv**2                # 1/r^12
        factor = np.divide(2 * distancePowTwelveInv - distancePowSixInv, distanceSqrd)            # (2/r^12 - 1/r^6)/r^2
        factor[factor == np.inf] = 0
        force = 24 * np.einsum('i,ij->ij', factor, dr)

        # force
        forceMatrix = np.zeros((npar, npar, ndim))
        upperTri = np.triu_indices(npar, 1)
        self.index = np.array(upperTri).T
        index = self.index[indices].T
        forceMatrix[(index[0], index[1])] = force
        forceMatrix[(index[1], index[0])] = -force

        return np.sum(forceMatrix, axis=1)

    def eval_acc_energy(self, r):
        """
        Evaluate acceleration and energy
        """
        npar = len(r)
        ndim = len(r[0])
        distanceSqrdAll, distanceSqrd, dr, indices = self.distance_matrix_triu(r)
        distancePowSixInv = np.nan_to_num(distanceSqrd**(-3))      # 1/r^6
        distancePowTwelveInv = distancePowSixInv**2                # 1/r^12
        
        factor = np.divide(2 * distancePowTwelveInv - distancePowSixInv, distanceSqrd)            # (2/r^12 - 1/r^6)/r^2
        factor[factor == np.inf] = 0
        force = 24 * np.einsum('i,ij->ij', factor, dr)

        # 
        forceTensor = np.zeros((npar, npar, ndim))
        upperTri = np.triu_indices(npar, 1)
        self.index = np.array(upperTri).T
        index = self.index[indices].T
        forceTensor[(index[0], index[1])] = force
        forceTensor[(index[1], index[0])] = -force

        acc = np.sum(forceTensor, axis=1)
        energy = np.sum(4 * (distancePowTwelveInv - distancePowSixInv - self.cutoff_corr))

        return acc, energy

    def distance_vector_par(self, r, i):
        """
        Find distance between a particle i and all other particles
        """
        dr = r - r[i]
        distanceVector = np.einsum('ij,ij->i', dr, dr)
        return dr, distanceVector

    def eval_energy_par(self, r, i):
        """
        Evaluate potential energy of particle i
        """
        _, distanceVector = self.distance_vector_par(r, i)
        distancePowSixInv = np.nan_to_num(distanceSqrd**(-3))
        distancePowTwelveInv = distancePowSixInv**2
        energy = np.sum(4 * (distancePowTwelveInv - distancePowSixInv - self.cutoff_corr))
        return energy

    def eval_acc_par(self, r, i):
        """
        Evaluate force on particle i
        """
        dr, distanceVectorSqrd = self.distance_vector_par(r, i)
        distancePowSixInv = np.nan_to_num(distanceVectorSqrd**(-3))
        distancePowTwelveInv = distancePowSixInv**2
        
        factor = np.divide(2 * distancePowTwelveInv - distancePowSixInv, distanceVectorSqrd)
        factor[factor == np.inf] = 0
        force = 24 * np.einsum('i,ij->ij', factor, dr)

        return -np.sum(force, axis=0)

    def eval_acc_energy_par(self, r, i):
        """
        Evaluate force and energy on particle i
        """
        dr, distanceVector = self.distance_vector_par(r, i)
        distancePowSixInv = np.nan_to_num(distanceSqrd**(-3))
        distancePowTwelveInv = distancePowSixInv**2

        energy = np.sum(4 * (distancePowTwelveInv - distancePowSixInv - self.cutoff_corr))
        
        factor = np.divide(2 * distancePowTwelveInv - distancePowSixInv, distanceVector)
        factor[factor == np.inf] = 0
        force = 24 * np.einsum('i,ij->ij', factor, dr)

        return np.sum(force, axis=1), energy

if __name__ == "__main__":
    r = np.random.random((10, 3))
    print(r)

    ff = LennardJones(1, 1, 3)
    a, u = ff.eval_acc_energy(r)
    print(a)

    for i in range(10):
        ai = ff.eval_acc_par(r, i)
        print(ai)
