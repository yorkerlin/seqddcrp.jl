push!(LOAD_PATH, "../src2/")
using Distributions
using Compat
using ConjugatePriors
import ConjugatePriors.NormalWishart

immutable DiagonalNormalWishart <: Distribution
    dim::Int
	factor::Array{NormalWishart}

    function DiagonalNormalWishart(mu::Vector{Float64}, kappa::Vector{Float64},
                                  beta::Vector{Float64}, alpha::Vector{Float64})
        d = length(mu)
		@assert d == length(beta)
		@assert d == length(alpha)
		@assert d == length(kappa)
	
        zmean::Bool = true
		factor=Array(NormalWishart, d)
        for i = 1:d
            factor[i]=NormalWishart([mu[i]],kappa[i],diagm(1.0/(2.0*beta[i])),2.0*alpha[i])
        end
        @compat new(d, factor)
    end
end
