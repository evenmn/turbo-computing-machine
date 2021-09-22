import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path


class TmpName:

    from .dump import Dump
    from .thermo import Thermo
    from .initvelocity import Zero
    from .forcefield import LennardJones
    from .integrator import VelocityVerlet
    from .sampler import BruteForce

    def __init__(self, dir, position, velocity=Zero(), info=False):
        self.p = Path(dir)
        self.p.mkdir(parents=True, exist_ok=True)

        self.r = position()
        self.v = velocity(self.r.shape)
        self.t = 0

        self.npar, self.ndim = self.r.shape

        self.dumpobj = self.Dump(np.inf, "dump.xyz", ())
        self.thermoobj = self.Thermo(np.inf, "log.tmp_name", ())
        self.outputs = []

        self.info = info

        self.forcefield = self.LennardJones(1, 1, 3)
        self.a, self.u = self.forcefield.eval_acc_energy(self.r)
        self.integrator = self.VelocityVerlet(dt=0.01)
        self.integrator.set_forcefield(self.forcefield)
        self.sampler = self.BruteForce(dx=0.01)

    def set_forcefield(self, forcefield):
        """
        Set forcfield

        forcefield : obj
            ForceField object from tmp_name.forcefield
        """
        self.a, self.u = forcefield.eval_acc_energy(self.r)
        self.forcefield = forcefield
        self.integrator.set_forcefield(self.forcefield)

    def set_integrator(self, integrator):
        """
        """
        self.integrator = integrator
        self.integrator.set_forcefield(self.forcefield)

    def set_sampler(self, sampler):
        self.sampler = sampler

    def set_output(self, style, filename):
        """
        Printing output to file
        """
        self.outputs.append(style)

    def dump(self, freq, file, *quantities):
        """Dump per-atom quantities to file
        """
        if self.info:
            print(f"\nDumping every {freq}th (", ", ".join(quantities), f") to file '{file}'")
        self.dumpobj = self.Dump(freq, file, quantities)

    def thermo(self, freq, file, *quantities):
        """Print thermo-quantities to file
        """
        if self.info:
            print(f"\nPrinting every {freq}th (", ", ".join(quantities), f") to file '{file}'")
        if "poteng" in quantities:
            self.compute_poteng = True
        else:
            self.compute_poteng = False
        self.thermoobj = self.Thermo(freq, file, quantities)

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

    def iterations(self, steps, out):
        self.t0 = self.t
        iterations = range(self.t0, self.t0 + steps + 1)
        if out == "tqdm":
            sys.stdout.flush()
            iterations = tqdm(iterations)
        elif out == "log":
            self.thermoobj.write_header()
        #else: whatever else will give no output ("no", "off", "false" etc)
        return iterations
        

    def run_md(self, steps, out="tqdm"):
        """
        """
        for self.t in self.iterations(steps, out):
            self.r, self.v, self.a, self.u = self.integrator(self.r, self.v, self.a)
            self.dumpobj(self)
            log = self.thermoobj(self)
            if out == "log":
                print(log, end="")

    def run_mc(self, steps, out="tqdm"):
        """
        """
        naccept = 0
        for self.t in self.iterations(steps, out):
            r_new = self.sampler.propose_move(self.r)
            u_new = self.forcefield.eval_energy(r_new)
            accept = self.sampler.accept_move(self.u, u_new)
            if accept:
                self.r = r_new
                self.u = u_new
                

                naccept += 1
            self.acc_ratio = naccept/(self.t-self.t0+1)

            self.dumpobj(self)
            log = self.thermoobj(self)
            if out == "log":
                print(log, end="")
