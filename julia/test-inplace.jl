using Revise, Metal, BenchmarkTools, Debugger

N = 10_000_000;
a = rand(Float32, N);
Ma = MtlArray(a);
r = Metal.zeros(Float32, 1);
