using ProgressMeter
using CSV
using DataFrames

open("data/Xtr0.csv") do file
    global lines = readlines(file)
end
popfirst!(lines)

function S(a::Char, b::Char)
    return 10 * convert(Float64, a==b) - 9
end

beta = 1
d = 1
e = 10

function K(x::String, y::String)
    n::Int64 = length(x)
    m::Int64 = length(y)
    M::Array{Float64, 2} = zeros(Float64, n, m)
    X::Array{Float64, 2} = zeros(Float64, n, m)
    Y::Array{Float64, 2} = zeros(Float64, n, m)
    X2::Array{Float64, 2} = zeros(Float64, n, m)
    Y2::Array{Float64, 2} = zeros(Float64, n, m)
    for j in 1:m
        for i in 1:n
            if (i == 1) || (j==1)
                M[i, j] = exp(beta * S(x[i], y[j]))
            else
                M[i, j] = exp(beta * S(x[i], y[j])) * (
                    1 + X[i-1, j-1] + Y[i-1, j-1] + M[i-1, j-1]
                )
                X[i, j] = exp(beta * d) * M[i-1, j] + exp(beta * e) * X[i-1, j]
                X2[i, j] = M[i-1, j] + X2[i-1, j]
            end
        end
        if (j > 1)
            Y[:, j] = exp(beta * d) * (M[:, j-1] + X[:, j-1]) + exp(beta * e) * Y[:, j-1]
            Y2[:, j] = M[:, j-1] + X2[:, j-1] + Y2[:, j-1]
        end
    end
    return log(1 + X2[n, m] + Y2[n, m] + M[n, m]) / beta
end

K(lines[1], lines[1])

L = length(lines[1:10])
gram = zeros(Float64, L, L)
@showprogress for a in 1:L, b in 1:a
    gram[a, b] = K(lines[a], lines[b])
end
df = DataFrame(gram)
CSV.write("LA_0_gram.csv", df)


L

df

gram

K(lines[1], lines[1])
