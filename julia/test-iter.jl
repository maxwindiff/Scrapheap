using Revise, Metal, BenchmarkTools

n = 1024 * 10000
a = rand(Float16, n)
b = rand(Float16, n)
c = zeros(Float16, n)
da = MtlArray(a)
db = MtlArray(b)
dc = MtlArray(c)

reset() = fill!(dc, 0)
check() = Array(dc) == Array(da) + Array(db)

# julia> @benchmark cpu()
# BenchmarkTools.Trial: 2048 samples with 1 evaluation.
#  Range (min … max):  1.649 ms … 10.710 ms  ┊ GC (min … max):  0.00% … 55.43%
#  Time  (median):     1.891 ms              ┊ GC (median):     0.00%
#  Time  (mean ± σ):   2.433 ms ±  1.277 ms  ┊ GC (mean ± σ):  15.66% ± 18.91%
#   ██▆▃▁      ▅▄▄▁▁   ▂▂▂                                     ▁
#   █████▅▁▄▄▄▁█████▇▇▇███▆▁▁▁▄▁▁▁▄▁▁▄▁▄▄▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▄▁▄▁▁▇ █
#   1.65 ms      Histogram: log(frequency) by time     9.71 ms <
#  Memory estimate: 19.53 MiB, allocs estimate: 2.
function cpu()
  global c = a + b
end

# julia> @benchmark dotplus()
# BenchmarkTools.Trial: 5151 samples with 1 evaluation.
#  Range (min … max):  830.583 μs …   2.488 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     930.208 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   968.349 μs ± 103.681 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%
#              ▄▇█▅▃
#   ▂▁▁▂▂▂▂▂▃▅▇█████▇▆▄▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▂▃▃▃▄▄▄▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂ ▃
#   831 μs           Histogram: frequency by time         1.25 ms <
#  Memory estimate: 6.12 KiB, allocs estimate: 204.
function dotplus()
  Metal.@sync begin
    global dc .= da .+ db
  end
end

# julia> @benchmark bcast()
# BenchmarkTools.Trial: 5308 samples with 1 evaluation.
#  Range (min … max):  827.166 μs …  3.243 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     921.333 μs              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   939.782 μs ± 81.692 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%
#           ▁▂▄▆▇██▇▆▅▄▃▃▃▂▂▂▁▂▁▂▁▁▁ ▁▁                          ▂
#   ▄▄▃▄▅▆▆█████████████████████████████▇▇██▇▇▆▆▄▄▅▆▆▄▆▅▃▃▁▅▅▆▃▄ █
#   827 μs        Histogram: log(frequency) by time      1.22 ms <
#  Memory estimate: 6.05 KiB, allocs estimate: 202.
function bcast()
  Metal.@sync begin
    broadcast!(+, dc, da, db)
  end
end

# julia> @benchmark kernel()
# BenchmarkTools.Trial: 6866 samples with 1 evaluation.
#  Range (min … max):  632.875 μs …  2.401 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     715.625 μs              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   725.804 μs ± 56.779 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%
#              ▁▃▄▅▆▇▇███▇▇▆▅▅▄▃▂▂▂▁▁▁ ▁▁▁▁ ▁▁                   ▃
#   ▃▁▄▃▁▃▃▁▄▅▇████████████████████████████████████▇▇█▇▇▇▇▇▇▇▇▇▇ █
#   633 μs        Histogram: log(frequency) by time       881 μs <
#  Memory estimate: 4.48 KiB, allocs estimate: 169.
function add(a::MtlDeviceArray{T}, b::MtlDeviceArray{T}, c::MtlDeviceArray{T}) where {T}
  i = thread_position_in_grid_1d()
  @inbounds c[i] = a[i] + b[i]
  return
end

function kernel()
  Metal.@sync begin
    @metal threads=1024 grid=10000 add(da, db, dc)
  end
end
