using Revise, Metal, BenchmarkTools, Combinatorics, Printf

function init(dims...; val=Float32(1))
  a = fill(val, dims...)
  return (a, MtlArray(a))
end

a, da = init(8192 * 8192);
b, db = init(8192, 8192);
c, dc = init(256, 512, 512);

function trygrains(f)
  # for grain in [1 2 4 8 16]
  #   @printf "grain=%-2d:" grain
  #   Metal.set_grain_size!(grain)  # temp hack to override the grain size
  #   f(grain)
  # end
  f(0)
end

function bench_p2(typ)
  d1 = MtlArray(fill(convert(typ, 1), 8192 * 8192))
  d2 = MtlArray(fill(convert(typ, 1), 8192, 8192))
  d3 = MtlArray(fill(convert(typ, 1), 256, 512, 512))
  @info typ typeof(d1) typeof(d2) typeof(d3) sum(d1) sum(d2) sum(d3)
  println()

  @printf "%-40s" "sum($(size(d1)))"
  trygrains() do _ @btime Metal.@sync sum($d1) end

  @printf "%-40s" "sum($(size(d2)))"
  trygrains() do _ @btime Metal.@sync sum($d2) end

  for dims in powerset(1:3)
    isempty(dims) && continue
    @printf "%-40s" "sum($(size(d3)); dims=$dims)"
    trygrains() do _ @btime Metal.@sync sum($d3; dims=$dims) end
  end

  println()
end

function bench_nonp2(typ)
  d1 = MtlArray(fill(convert(typ, 1), 8191 * 8191))
  d2 = MtlArray(fill(convert(typ, 1), 8191, 8191))
  d3 = MtlArray(fill(convert(typ, 1), 26, 237, 10888))  # roughly 8191*8191
  @info typ typeof(d1) typeof(d2) typeof(d3) sum(d1) sum(d2) sum(d3)
  println()

  @printf "%-40s" "sum($(size(d1)))"
  trygrains() do _ @btime Metal.@sync sum($d1) end

  @printf "%-40s" "sum($(size(d2)))"
  trygrains() do _ @btime Metal.@sync sum($d2) end

  for dims in powerset(1:3)
    isempty(dims) && continue
    @printf "%-40s" "sum($(size(d3)); dims=$dims)"
    trygrains() do _ @btime Metal.@sync sum($d3; dims=$dims) end
  end

  println()
end

function benchall()
  for typ in [Int64 Float32 Int32 Float16 Int16 Int8]
    println("===================== $typ (power of 2) =====================\n")
    bench_p2(typ)
    println("===================== $typ (non-pow of 2) =====================\n")
    bench_nonp2(typ)
  end
end

neutral_element(op, T) =
    error("""GPUArrays.jl needs to know the neutral element for your operator `$op`.
             Please pass it as an explicit argument to `GPUArrays.mapreducedim!`,
             or register it globally by defining `GPUArrays.neutral_element(::typeof($op), T)`.""")
neutral_element(::typeof(Base.:(|)), T) = zero(T)
neutral_element(::typeof(Base.:(+)), T) = zero(T)
neutral_element(::typeof(Base.add_sum), T) = zero(T)
neutral_element(::typeof(Base.:(&)), T) = one(T)
neutral_element(::typeof(Base.:(*)), T) = one(T)
neutral_element(::typeof(Base.mul_prod), T) = one(T)
neutral_element(::typeof(Base.min), T) = typemax(T)
neutral_element(::typeof(Base.max), T) = typemin(T)

@inline function reduce_warp(op, val)
  offset = 0x00000001
  while offset < 32
      val = op(val, simd_shuffle_down(val, offset))
      offset <<= 1
  end
  return val
end

#@inline reduce_warp(::typeof(Base.:(+)), val) = simd_sum(val)

function reduce_group(op, neutral::T, in::MtlDeviceArray{T}, out::MtlDeviceArray{T}, ::Val{stride}) where {T, stride}
  tid = thread_position_in_threadgroup_1d()
  blockSize = threads_per_threadgroup_1d()
  shared = MtlThreadGroupArray(T, 1024)

  @inbounds begin
    # Read and reduce multiple values per thread
    val = 0
    base = (thread_position_in_grid_1d() - 1) * stride
    i = base + 1
    while i <= base+stride
      val = op(val, in[i])
      i += 1
    end
    shared[tid] = val
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)

    offset::UInt32 = 512
    while offset > 16
      if blockSize >= 2 * offset
        if tid <= offset
          shared[tid] += shared[tid + offset]
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
      end
      offset >>= 1
    end

    if simdgroup_index_in_threadgroup() == 1
      shared[tid] = reduce_warp(op, shared[tid])
    end

    if tid == 1
      out[threadgroup_position_in_grid_1d()] = shared[tid]
    end
  end

  return
end

function reduce_coop(op, neutral::T, in::MtlDeviceArray{T}, out::MtlDeviceArray{T}, ::Val{stride}) where {T, stride}
  # shared mem for partial sums
  shared = MtlThreadGroupArray(T, 32)

  wid  = simdgroup_index_in_threadgroup()
  lane = thread_index_in_simdgroup()

  val = op(neutral, neutral)

  @inbounds begin
    # Read and reduce multiple values per thread
    base = (thread_position_in_grid_1d() - 1) * stride
    i = base + 1
    while i <= base+stride
      val = op(val, in[i])
      i += 1
    end
  end

  # each warp performs partial reduction
  val = reduce_warp(op, val)

  # write reduced value to shared memory
  if lane == 1
    @inbounds shared[wid] = val
  end
  threadgroup_barrier(Metal.MemoryFlagThreadGroup)

  # read from shared memory only if that warp existed
  val = if thread_index_in_threadgroup() <= fld1(threads_per_threadgroup_1d(), 32)
    @inbounds shared[lane]
  else
    neutral
  end

  # final reduce within first warp
  if wid == 1
    out[threadgroup_position_in_grid_1d()] = reduce_warp(op, val)
  end

  return
end

function myreduce(op, a::MtlArray{T}, stride=4) where {T}
  @assert length(a) % stride == 0 "Not suppported yet"

  threads = min(length(a), 1024)
  groups = cld(length(a), 1024 * stride)
  b = similar(a, groups)
  @metal threads=threads grid=groups reduce_group(op, neutral_element(op, T), a, b, Val(stride))
  #@metal threads=threads grid=groups reduce_coop(op, neutral_element(op, T), a, b, Val(stride))
  if groups == 1
    return b[1]
  elseif groups < 32
    return sum(b)
  else
    return myreduce(op, b)
  end
end

mysum(a, stride=4) = myreduce(+, a, stride)
