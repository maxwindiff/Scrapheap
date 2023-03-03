using Revise, Metal, BenchmarkTools
typ = Float32

function load(a::MtlDeviceArray{T}, origin_a=(1, 1)) where {T}
    sg_a = simdgroup_load(a, origin_a)
    return
end

function load_tg(a::MtlDeviceArray{T}) where {T}
    tg = MtlThreadGroupArray(T, (8, 8))
    pos = thread_position_in_threadgroup_2d()
    tg[pos.x, pos.y] = a[pos.x, pos.y]

    sg_a = simdgroup_load(tg)
    return
end

function load_store(a::MtlDeviceArray{T}, b::MtlDeviceArray{T},
                    origin_a=(1, 1), origin_b=(1, 1)) where {T}
    sg_a = simdgroup_load(a, origin_a)
    simdgroup_store(sg_a, b, origin_b)
    return
end

function load_store_tg(a::MtlDeviceArray{T}, b::MtlDeviceArray{T}) where {T}
    pos = thread_position_in_threadgroup_2d()

    tg_a = MtlThreadGroupArray(T, (8, 8))
    tg_a[pos.x, pos.y] = a[pos.x, pos.y]
    sg_a = simdgroup_load(tg_a)

    tg_b = MtlThreadGroupArray(T, (8, 8))
    simdgroup_store(sg_a, tg_b)
    b[pos.x, pos.y] = tg_b[pos.x, pos.y]

    return
end

function mul(a::MtlDeviceArray{T}, b::MtlDeviceArray{T}, c::MtlDeviceArray{T}) where {T}
    sg_a = simdgroup_load(a)
    sg_b = simdgroup_load(b)
    sg_c = simdgroup_multiply(sg_a, sg_b)
    simdgroup_store(sg_c, c)
    return
end

function mad(a::MtlDeviceArray{T}, b::MtlDeviceArray{T}, c::MtlDeviceArray{T}, d::MtlDeviceArray{T}) where {T}
    sg_a = simdgroup_load(a)
    sg_b = simdgroup_load(b)
    sg_c = simdgroup_load(c)
    sg_d = simdgroup_multiply_accumulate(sg_a, sg_b, sg_c)
    simdgroup_store(sg_d, d)
    return
end

a = MtlArray(rand(typ, 8, 8))
b = MtlArray(rand(typ, 8, 8))
c = MtlArray(rand(typ, 8, 8))
d = MtlArray(zeros(typ, 8, 8))

#@device_code_warntype @metal threads=(8,8) load_store_tg(a, b)
