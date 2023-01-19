# optimism as a function of num splits
library(RcppEigen)
library(ggplot2)

#MSE Loss and derivatives
loss <- function(y,y.hat){
    mean((y-y.hat)^2)
}
dloss <- function(y, y.hat) -2*sum(y-y.hat)
ddloss <- function(y, y.hat) 2*length(y)

# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Compile cpp and load into R environment
Rcpp::sourceCpp("approximate_bayesian_tree_update.cpp")


##### test if tree actually can train ####
## A simple gtb.train example with linear regression:
set.seed(123)
n <- 500
x <- as.matrix(runif(n, 0, 4), ncol=1)
y <- rnorm(n, x, 1)
#x.test <- runif(500, 0, 4)
#y.test <- rnorm(500, x.test, 1)
# compute derivatives
y0.hat <- mean(y)
g <- sapply(y, dloss, y.hat=y0.hat)
h <- sapply(y, ddloss, y.hat=y0.hat)

# Train gbtree
cir_sim <- cir_sim_mat(100,100)
tree <- new(GBTREE)
tree$train(g, h, x, 
           cir_sim, 
           FALSE, 1.0, 1
           )

y.pred <- y0.hat + tree$predict_data(x)

plot(x, y)
points(x, y.pred, col="red")


##### extract triplet [pred, var(w), var(y)] 

