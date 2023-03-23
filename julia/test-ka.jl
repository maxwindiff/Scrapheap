using KernelAbstractions, Metal, Metal.MetalKernels, Adapt, Random

Backend = MetalBackend;

backend = Backend();
x = allocate(backend, Float32, 5);
@show adapt(CPU(), x) isa Array
y = adapt(backend, Array{Float32}(undef, 5));
@show typeof(y) == typeof(x)

M = 1024;
ET = KernelAbstractions.supports_float64(backend) ? Float64 : Float32;

A = rand!(allocate(backend, ET, M));
B = rand!(allocate(backend, ET, M));

a = Array{ET}(undef, M);
KernelAbstractions.copyto!(backend, a, B);
KernelAbstractions.copyto!(backend, A, a);
KernelAbstractions.synchronize(backend);

@show isapprox(a, Array(A))
@show isapprox(a, Array(B))
