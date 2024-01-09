 
  
from setuptools import setup

d = {}
exec(open("ghibtools/version.py").read(), None, d)
version = d['version']

install_requires = [
                    'numpy',
                    'scipy',
                    'pandas',
                    'xarray',
                    'matplotlib',
                    'scikit-learn',
                    'neurokit2',
                    'seaborn',
                    'pingouin',
                    ]

long_description = ""

setup(
    name = "ghibtools",
    version = version,
    packages = ['ghibtools'],
    install_requires=install_requires,
    author = "V. Ghibaudo",
    author_email = "valentin.ghibaudo@inserm.fr",
    description = "Toolbox from Valentin Ghibaudo, Neuroscience PhD Student",
    long_description = long_description,
    license = "MIT",
    url='https://github.com/ValentinGhibaudo/ghibtools',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering']
)
