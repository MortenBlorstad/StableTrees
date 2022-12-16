
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
            os.path.join(os.path.dirname(__file__), "../include/thirdparty/pybind11"),
            os.path.join(os.path.dirname(__file__), "../include/thirdparty/eigen")
        ],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name='stabletrees',
    version='0.1.0',    
    description='A example Python package',
    author='Morten Blørstad',
    author_email='mblorstad@email.com',
    url="https://github.com/MortenBlorstad/StableTrees",
    package_dir={"": "src"},
    packages=find_packages(where="./src"),
    ext_modules=ext_modules,
    install_requires=['numpy'],
    python_requires=">=3.6",

    
)