import sys
import setuptools


long_description = '''
SpyRL is an RL framework
'''

if sys.version_info < (3, 6):
    sys.exit('Python>=3.6 is required.')

setuptools.setup(
    name="spyrl",
    version="0.2.4",
    url="https://github.com/budi-kurniawan/spyrl",

    description="Reinforcement Learning framework",
    long_description=long_description,
    license='Apache License Version 2.0',

    packages=setuptools.find_packages(),
    platforms='any',

    install_requires=[
        'gym[all]>=0.25.2',
        'matplotlib==3.4.0',
        'pygame',
        'torch',
        'scipy', 'mpi4py'
    ],
    extras_require={
    },
    package_data={
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)