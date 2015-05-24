push!(LOAD_PATH, "../src/")
push!(LOAD_PATH, "../src2/")

using seqddCRP
using Distributions
using ConjugatePriors
import ConjugatePriors.NormalWishart

#out = open("5.csv", "a") 

x = readdlm("./balley.csv", ',')
x = x'

N = size(x, 2)
p = size(x, 1)
prior = NormalWishart(zeros(2), 1e-7, eye(2) / 4, 4.0001)

F = crp_F(N, 1e-1)

crp, predictive_likelihood = var_gaussian_mixture(x, F, prior, singular_init_dist)
ll = infer(crp, 100)

#println(ll)
