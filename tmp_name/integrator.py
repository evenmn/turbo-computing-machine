
class Integrator:
    """
    Integrator base class
    """
    def __init__(self, dt, forcefield):
        self.dt = dt
        self.forcefield = forcefield


class Euler(Integrator):
    """
    Forward Euler integrator
    """
    def __call__(self, r, v, a):
        r_ = r + v * self.dt
        v_ = v + a * self.dt
        #r_ = self.boundary.check_position(r_)
        #v_ = self.boundary.check_velocity(v_)
        a_ = self.forcefield.eval_acc(r_)
        return r_, v_, a_


class EulerCromer(Integrator):
    """
    Euler-Cromer integrator
    """
    def __call__(self, r, v, a):
        v_ = v + a * self.dt
        r_ = r + v_ * self.dt
        #r_ = self.boundary.check_position(r_)
        #v_ = self.boundary.check_velocity(v_)
        a_ = self.forcefield.eval_acc(r_)
        return r_, v_, a_


class VelocityVerlet(Integrator):
    """
    Velocity Verlet integrator
    """
    def __call__(self, r, v, a):
        r_ = r + v * self.dt + 0.5 * a * self.dt**2
        #r_ = self.boundary.check_position(r_)
        a_ = self.forcefield.eval_acc(r_)
        v_ = v + 0.5 * (a_ + a) * self.dt
        #v_ = self.boundary.check_velocity(v_)
        return r_, v_, a_
