import numpy as np
from pathlib import Path


class TmpName:

    from .dump import Dump
    from .initvelocity import Zero

    def __init__(self, dir, position, velocity=Zero(), info=False):
        self.p = Path(dir)
        self.p.mkdir(parents=True, exist_ok=True)

        self.r = position()
        self.v = velocity(self.r.shape)

        self.npar, self.ndim = self.r.shape

        self.dumpobj = self.Dump(np.inf, "dump.xyz", ())
        self.outputs = []

        self.info = info


    def set_forcefield(self, forcefield):
        """
        Set forcfield

        forcefield : obj
            ForceField object from tmp_name.forcefield
        """
        self.a = forcefield.eval_acc(self.r)
        self.u = forcefield.eval_energy(self.r)
        self.forcefield = forcefield

    def set_output(self, style, filename):
        """
        Printing output to file
        """
        self.outputs.append(style)

    def set_motion(self, motion):
        """
        Set motion (MD or MC)
        """
        self.motion = motion

    def snapshot(self, filename, vel=False):
        """Take snapshot of system and write to xyz-file
        """
        if self.info:
            print(f"\nSnapshot saved to file '{filename}'")
        if vel:
            lst = ('x', 'y', 'z', 'vx', 'vy', 'vz')
        else:
            lst = ('x', 'y', 'z')
        tmp_dumpobj = self.Dump(1, filename, lst[:self.ndim])
        tmp_dumpobj(self)
        del tmp_dumpobj

    def run(self, steps):
        self.r, self.v, self.a = self.motion.run(steps, self.r, self.v, self.a, self.u)
