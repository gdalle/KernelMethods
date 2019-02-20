open("data/Xtr0.csv") do file
    global lines = readlines(file)
end
popfirst!(lines)

function S(a::Char, b::Char)
    return convert(Float64, a==b)
end

beta = 1.
d = 1.
e = 1.

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
    return 1 + X2[n, m] + Y2[n, m] + M[n, m]
end
