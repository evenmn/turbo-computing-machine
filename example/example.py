from tmp_name import TmpName
from tmp_name.motion import MC, MD
from tmp_name.forcefield import LennardJones
from tmp_name.initposition import FCC
from tmp_name.initvelocity import Temperature

tn = TmpName("test", FCC(4, 10, 3), Temperature(1.4))
tn.snapshot("initial.xyz")
tn.set_forcefield(LennardJones(1, 1, 3))
tn.set_motion(MD(dt=0.01))
tn.run(steps=100)
tn.snapshot("final.xyz")

