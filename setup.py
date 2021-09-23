from setuptools import setup

setup(name='tmp_name',
      version='0.0.1',
      description='Simple Monte Carlo and Molecuar Dynamics simulator',
      url='http://github.com/evenmn/turbo-computing-machine',
      author='Even Marius Nordhagen',
      author_email='evenmn@mn.uio.no',
      license='MIT',
      packages=['tmp_name'],
      install_requires=["numpy", "tqdm"],
      zip_safe=False)
