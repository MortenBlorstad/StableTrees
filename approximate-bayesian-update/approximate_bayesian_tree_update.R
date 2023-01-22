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
n <- 1000
x <- as.matrix(runif(n, 0, 4), ncol=1)
#x <- as.matrix(sample(1:4, size=n, replace = TRUE))
y <- rnorm(n, x, 2)
#x.test <- runif(500, 0, 4)
#y.test <- rnorm(500, x.test, 1)
# compute derivatives
y0.hat <- mean(y)
g <- sapply(y, dloss, y.hat=y0.hat)
h <- sapply(y, ddloss, y.hat=y0.hat)

# Train gbtree
cir_sim <- cir_sim_mat(100,100)
tree_prior <- new(GBTREE)
tree_prior$train(g, h, x, 
           cir_sim, 
           FALSE, 1.0, 1
           )

y.pred <- y0.hat + tree_prior$predict_data(x)
par(mfrow=c(1,4))
plot(x, y, main="Prior tree", sub = "Trained only on dataset 1")
points(x, y.pred, col="red")

##### extract triplet [pred, var(w), var(y)] 
if(F){
    y.pred.var <- tree_prior$prediction_variance(x)
    y.var <- tree_prior$response_variance(x)
    w.weight <- tree_prior$prediction_weight(x)
    par(mfrow=c(1,4))
    plot(x, y, main="pred")
    points(x, y.pred, col="red")
    plot(x, y.pred.var, main="w.var")
    plot(x, y.var, main="y.var")
    plot(x, w.weight, main="ratio")
    par(mfrow=c(1,1))
}

##### generate data ####
set.seed(321)
n2 <- 100
x2 <- as.matrix(runif(n2, 0, 4), ncol=1)
y2 <- rnorm(n2, x2, 1)
g2 <- sapply(y2, dloss, y.hat=y0.hat)
h2 <- sapply(y2, ddloss, y.hat=y0.hat)
tree_d2 <- new(GBTREE)
tree_d2$train(g2, h2, x2, 
                 cir_sim, 
                 FALSE, 1.0, 1
)
y.pred2 <- y0.hat + tree_d2$predict_data(x2)
plot(x2, y2, main="Tree 2", sub = "Trained only on dataset 2")
points(x2, y.pred2, col="red")


##### build posterior tree #####
set.seed(456)
n1 <- n
# sample prior
xb <- as.matrix(sample(x2, size=n1, replace = T))
wb <- tree_prior$predict_data(xb)
gamma_b <- tree_prior$prediction_weight(xb)
hb <- 2*gamma_b
gb <- - hb * wb
# use complete square to train
x_ <- rbind(x2, xb)
g_ <- c(g2, gb)
h_ <- c(h2, hb)
tree_posterior <- new(GBTREE)
tree_posterior$train(
    g_, h_, x_, 
   cir_sim, 
   FALSE, 1.0, 1
)
y.pred_posterior <- y0.hat + tree_posterior$predict_data(x2)
plot(x2, y2, main="Posterior tree", sub="Trained on dataset 2 & distributional information from prior tree")
points(x2, y.pred_posterior, col="red")

##### build optimal tree ####
x_1_2 <- rbind(x, x2)
g_1_2 <- c(g, g2)
h_1_2 <- c(h, h2)
tree_opt <- new(GBTREE)
tree_opt$train(
    g_1_2, h_1_2, x_1_2,
    cir_sim, 
    FALSE, 1.0, 1
)
y.pred_opt <- y0.hat + tree_opt$predict_data(x_1_2)
plot(x_1_2, c(y,y2), main="Optimal tree", sub="Trained on dataset 1 & 2")
points(x_1_2, y.pred_opt, col="red")


