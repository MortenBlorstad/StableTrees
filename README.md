# StableTrees

|                                   | Baseline        | NU             | TR             | SL             | ABU            | BABU          |
|-----------------------------------|-------|-------|-------|-------|-------|-------|
| **Panel A: loss (MSE)**           |
| Boston                            | 21.326 (1.00)   | 22.874 (1.07)  | 22.874 (1.07)  | 21.879 (1.03)  | 20.975 (0.98)  | **18.813 (0.88)** |
| Carseats                          | 6.512 (1.00)    | 6.937 (1.07)   | 6.937 (1.07)   | 6.823 (1.05)   | **6.507 (1.00)** | 6.531 (1.00)  |
| College                           | **2801700.660 (1.00)** | 3420547.411 (1.22) | 3329944.881 (1.19) | 3192454.561 (1.14) | 2918508.528 (1.04) | 2886392.353 (1.03) |
| Hitters                           | 120557.157 (1.00) | 123574.015 (1.03) | 128804.004 (1.07) | 122364.386 (1.01) | 116871.759 (0.97) | **113851.957 (0.94)** |
| Wage                              | 1214.536 (1.00) | 1233.701 (1.02) | 1233.701 (1.02) | 1232.200 (1.01) | **1211.076 (1.00)** | 1216.604 (1.00) |
| **Panel B: $S = mean((T1(x)-T2(x))^2)$** |
| Boston                            | 11.151 (1.00)   | **0.722 (0.06)** | 0.722 (0.06)   | 1.224 (0.11)   | 10.135 (0.91)  | 6.873 (0.62)  |
| Carseats                          | 2.455 (1.00)    | 0.147 (0.06)   | 0.147 (0.06)   | **0.084 (0.03)** | 1.967 (0.80)   | 0.749 (0.31)  |
| College                           | 1513887.425 (1.00) | **135456.170 (0.09)** | 846595.179 (0.56) | 161846.572 (0.11) | 1391364.344 (0.92) | 930181.108 (0.61) |
| Hitters                           | 55772.055 (1.00) | 4007.482 (0.07) | 12829.108 (0.23) | **226




## Installation

The StableTrees package is available on [PyPi](https://pypi.org/project/stabletrees/)

```
pip install stabletrees
```

## Developing
Clone the repository
```git
git clone https://github.com/MortenBlorstad/StableTrees.git
```

To build the Python-package, first make sure that Eigen is installed and available in the include path `include/thirdparty`.

```
cd include\thirdparty
git clone https://gitlab.com/libeigen/eigen.git
cd ../..
```
Then StableTrees is installable in editable mode by
```
pip install -e ./python-package
```
