using Revise, Metal, BenchmarkTools, StaticArrays

n = 16384;
nn = n * n;
a = MtlArray(rand(Float32, n, n));
b = MtlArray(rand(Float32, n, n));

function copy1d(a, b, ::Val{dims}) where {dims}
  I = @inbounds CartesianIndices(dims)[thread_position_in_grid_1d()]
  @inbounds a[I] = b[I]
  return
end

function run1d()
  @Metal.sync begin
    @metal threads=1024 grid=cld(nn, 1024) copy1d(a, b, Val(axes(a)))
  end
end

function run1d_static()
  @Metal.sync begin
    @metal threads=1024 grid=cld(nn, 1024) copy1d(
      # hardcode for now
      SizedArray{Tuple{16384, 16384}}(a),
      SizedArray{Tuple{16384, 16384}}(b), Val(axes(a)))
  end
end

function copy1d_manual(a, b)
  pos = thread_position_in_grid_1d() - 1
  r = (pos & 16383) + 1
  c = (pos >> 14) + 1
  @inbounds a[r,c] = b[r,c]
  return
end

function run1d_manual()
  @Metal.sync begin
    @metal threads=1024 grid=cld(nn, 1024) copy1d_manual(a, b)
  end
end

function copy2d(a, b)
  (i, j) = thread_position_in_grid_2d()
  @inbounds a[i,j] = b[i,j]
  return
end

function run2d()
  @Metal.sync begin
    @metal threads=(32,32) grid=(cld(n,32), cld(n,32)) copy2d(a, b)
  end
end

function broadcast()
  @Metal.sync begin
    global a .= b
  end
end

# @btime run2d() # optimal
# @btime broadcast(); # current a .= b
# @btime run1d()
# @btime run1d_manual()
