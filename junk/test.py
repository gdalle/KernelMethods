import julia
j = julia.Julia()

x = j.include("test.jl")
y = j.eval("double(7)")

print(x, y)
