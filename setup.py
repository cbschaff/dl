from setuptools import setup

setup(
    name='dl',
    version='0.1',
    description=
    'Deep Reinforcement Learning library based on PyTorch, OpenAI Gym, and gin config, and Tensorboard (for visualization and logging). The library is designed for research and fast iteration. It is highly modular to speed up the implementation of new algorithms and shorten iteration cycles.',
    url='https://github.com/cbschaff/dl.git',
    author='Chip Schaff',
    author_email='cbschaff@ttic.edu',
    license='MIT',
    packages=['dl'],
    install_requires=[
        'pyyaml==5.3.1',
        'gin-config==0.3.0',
    ],
    zip_safe=False)
