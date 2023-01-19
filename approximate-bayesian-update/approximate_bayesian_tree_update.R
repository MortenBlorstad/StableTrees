# optimism as a function of num splits

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(RcppEigen)
library(ggplot2)

Rcpp::sourceCpp("approximate_bayesian_tree_update.cpp")






