using Metal
using Flux
using BenchmarkTools
using Debugger

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

a = rand(Float32, 8, 8)
b = rand(Float32, 8, 8)
c = Dense(8 => 8)

da = gpu(a)
db = gpu(b)
dc = gpu(c)
