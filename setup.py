import setuptools

setuptools.setup(
    name="dl",
    version="0.0.1",
    author="Charles Schaff",
    author_email="cbschaff@ttic.edu",
    description="Deep Reinforcement Learning Code",
    url="https://github.com/cbschaff/dl",
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
      'numpy',
      'matplotlib',
      'imageio',
      'pandas',
      'gin-config',
      'gym',
      'gym[atari]',
      'gym[box2d]',
      'pyyaml==5.3.1'
    ],
    zip_safe=False,
)
