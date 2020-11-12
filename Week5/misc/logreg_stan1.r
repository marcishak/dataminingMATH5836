options(repos = c(getOption("repos"), rstan = "http://wiki.stan.googlecode.com/git/R")) install.packages('rstan', type = 'source')


library(rstan)
PoD_samples <- sampling(object = PoD_model, 
                        data = list(N = N, det = pod_df$det, depth = pod_df$depth,
                                    K = K, depth_pred = depth_pred), 
                        seed = 1008)
