from tmp_name import TmpName
from tmp_name.sampler import ImportanceSampling
from tmp_name.initposition import FCC
from tmp_name.initvelocity import Temperature

tn = TmpName("test", FCC(4, 10, 3), Temperature(1.4))
tn.snapshot("initial.xyz")

tn.dump(1, "dump.xyz", 'x', 'y', 'z')

# relax with molecular dynamics
tn.thermo(1, "md.log", 'step', 'time', 'poteng', 'kineng')
tn.run_md(steps=100)
tn.snapshot("after_md.xyz")

# run Monte Carlo
tn.set_sampler(ImportanceSampling(dx=1.0, dt=1.0, stillinger_lim=3.0))
tn.thermo(1, "mc.log", 'step', 'poteng', 'acc_ratio')
tn.run_mc(steps=1000)
tn.snapshot("final.xyz")
