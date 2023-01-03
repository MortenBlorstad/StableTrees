
import os
from glob import glob
from setuptools import find_packages, setup
import pybind11; print(pybind11.__file__)
from pybind11.setup_helpers import Pybind11Extension


__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "_stabletrees",
        sorted(glob("src/*.cpp")),
        cxx_std=14,
        include_dirs=[
            os.path.join(os.path.dirname(__file__), "../include/stabletrees"),
            os.path.join(os.path.dirname(__file__), "../include/stabletrees/trees"),
            os.path.join(os.path.dirname(__file__), "../include/stabletrees/splitters"),
            os.path.join(os.path.dirname(__file__), "../include/stabletrees/criterions"),
            os.path.join(os.path.dirname(__file__), "../include/stabletrees/optimism"),
            os.path.join(os.path.dirname(__file__), "../include/thirdparty/eigen"),
            "/usr/include/eigen3",
            "/usr/local/include/eigen3",
            "/usr/local/include/thirdparty/eigen",
            "../eigen",
        ],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name='stabletrees',
    version='0.1.0',    
    description='Regression tree with stable update',
    author='Morten Blørstad',
    author_email='mblorstad@email.com',
    url="https://github.com/MortenBlorstad/StableTrees",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    install_requires=['numpy', "scikit-learn", "matplotlib"],
    python_requires=">=3.6",

    
)