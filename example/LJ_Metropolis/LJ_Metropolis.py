from tmp_name import TmpName
from tmp_name.sampler import Metropolis
from tmp_name.moves import Trans, TransMH
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
tn.set_sampler(Metropolis())
tn.add_move(Trans(dx=0.01), 0.3)
tn.add_move(TransMH(dx=0.01, Ddt=0.01), 0.7)
tn.thermo(1, "mc.log", 'step', 'poteng', 'acc_ratio')
tn.run_mc(steps=1000, out='log')
tn.snapshot("final.xyz")
