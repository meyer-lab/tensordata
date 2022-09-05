from setuptools import setup, find_packages

setup(name='tensordata',
      version='0.0.5',
      description='A common repository for tensor structured datasets.',
      url='https://github.com/meyer-lab/tensordata',
      license='MIT',
      packages=find_packages(exclude=['doc']),
      install_requires=['numpy', 'tensorly'])
