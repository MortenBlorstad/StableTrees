from setuptools import setup, find_packages
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy
extensions = [
    Extension("stabletrees.cnode", ["stabletrees/cnode.pyx"], include_dirs=[numpy.get_include()]),
    # Add other extensions if needed
]
setup(
    name='stabletrees',
    version='0.1',
    description='Regression tree with stable update',
    author='Morten Blørstad',
    author_email='mblorstad@email.com',
    url="https://github.com/MortenBlorstad/StableTrees/StableTreesAllPython",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    python_requires=">=3.6",
   
)


