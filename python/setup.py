import os
from setuptools import find_packages, setup

from pybind11.setup_helpers import Pybind11Extension
from glob import glob
print("="*30)
print(os.path.join(os.path.dirname(__file__), "cpp"),__file__)
print("="*30)


ext_modules = [
    Pybind11Extension(
        "stable_trees",
        sorted(glob("cpp/*.cpp")),
        cxx_std=14,
        include_dirs=[
            os.path.join(os.path.dirname(__file__), "../cpp"),
            os.path.join(os.path.dirname(__file__), "../cpp/include/thirdparty/eigen/Eigen"),
            os.path.join(os.path.dirname(__file__), "../cpp/include/thirdparty/pybind11"),
        ]
    ),
]
print(os.path)

with open("../README.md") as f:
    long_description = f.read()

setup(
    name="StableTrees",
    version=1,
    author="Morten Blørstad",
    author_email="mblorstad@gmail.com",
    url="https://github.com/MortenBlorstad/StableTrees",
    description="Stable Trees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    install_requires=["numpy", "scipy"],
    packages=find_packages(where="python"),
    zip_safe=False,
    python_requires=">=3.6",
)