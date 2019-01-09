from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='dl',
      packages=[package for package in find_packages()
                if package.startswith('dl')],
      install_requires=[
          'numpy',
          'matplotlib',
          'imageio',
          'pandas',
          # 'pytorch',
          # 'torchvision',
          # 'mpi4py',
      ],
      description='deep learning code',
      author='Chip Schaff',
      author_email='cbschaff@ttic.edu',
      version='0.0')
