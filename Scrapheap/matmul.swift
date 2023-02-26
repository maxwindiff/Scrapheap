import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders

func testMatmul() {
    var M = 2048, N = M, K = M; // for convenience, keep all dimensions the same

    let device = MTLCreateSystemDefaultDevice()!
    let cmdQueue = device.makeCommandQueue()!

    let (bufA, mtxA) = makeRandomDeviceMtx(device, M, K)
    let (bufB, mtxB) = makeRandomDeviceMtx(device, K, N)
    let (bufC, mtxC) = makeRandomDeviceMtx(device, M, N)

    let mpsMatMul = MPSMatrixMultiplication(
        device: device, transposeLeft: false, transposeRight: false,
        resultRows: M, resultColumns: N, interiorColumns: K, alpha: 1, beta: 0)
    benchmark(name: "mps") {
        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        cmdBuffer.label = "MPSMatrixMultiplication"
        mpsMatMul.encode(commandBuffer: cmdBuffer, leftMatrix: mtxA, rightMatrix: mtxB, resultMatrix: mtxC)
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
    }
    let expectedC = device.makeBuffer(bytes: bufC.contents(), length: MemoryLayout<Float>.stride * M * N)!
    clearDeviceMtx(bufC)
    printIfSmall(bufA, bufB, bufC, M, N, K)

    let matmul2DThreadtile = makePipeline(device, "matmul_2d_threadtile")
    benchmark(name: "matmul_2d_threadtile (8x8, 4x4)") {
        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        cmdBuffer.label = "2d_threadtile"
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(matmul2DThreadtile)
        cmdEncoder.setBuffers([bufA, bufB, bufC], offsets: [0, 0, 0], range: 0..<3)
        cmdEncoder.setBytes(&M, length: MemoryLayout<Int>.stride, index: 3)
        cmdEncoder.setBytes(&N, length: MemoryLayout<Int>.stride, index: 4)
        cmdEncoder.setBytes(&K, length: MemoryLayout<Int>.stride, index: 5)
        cmdEncoder.dispatchThreads(MTLSizeMake(M/8, N/8, 1), threadsPerThreadgroup: MTLSizeMake(8, 8, 1))
        cmdEncoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
    }
    checkAndReset(want: expectedC, got: bufC)
    printIfSmall(bufA, bufB, bufC, M, N, K)

    let matmul1DThreadtile = makePipeline(device, "matmul_1d_threadtile")
    benchmark(name: "matmul_1d_threadtile (8x8, 1x4)") {
        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        cmdBuffer.label = "1d_threadtile"
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(matmul1DThreadtile)
        cmdEncoder.setBuffers([bufA, bufB, bufC], offsets: [0, 0, 0], range: 0..<3)
        cmdEncoder.setBytes(&M, length: MemoryLayout<Int>.stride, index: 3)
        cmdEncoder.setBytes(&N, length: MemoryLayout<Int>.stride, index: 4)
        cmdEncoder.setBytes(&K, length: MemoryLayout<Int>.stride, index: 5)
        cmdEncoder.dispatchThreads(MTLSizeMake(M, N/4, 1), threadsPerThreadgroup: MTLSizeMake(8, 8, 1))
        cmdEncoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
    }
    checkAndReset(want: expectedC, got: bufC)
    printIfSmall(bufA, bufB, bufC, M, N, K)

    let matmulThreadgroup = makePipeline(device, "matmul_threadgroup")
    benchmark(name: "matmul_threadgroup (16x16)") {
        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        cmdBuffer.label = "threadgroup"
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(matmulThreadgroup)
        cmdEncoder.setBuffers([bufA, bufB, bufC], offsets: [0, 0, 0], range: 0..<3)
        cmdEncoder.setBytes(&M, length: MemoryLayout<Int>.stride, index: 3)
        cmdEncoder.setBytes(&N, length: MemoryLayout<Int>.stride, index: 4)
        cmdEncoder.setBytes(&K, length: MemoryLayout<Int>.stride, index: 5)
        cmdEncoder.dispatchThreads(MTLSizeMake(M, N, 1), threadsPerThreadgroup: MTLSizeMake(16, 16, 1))
        cmdEncoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
    }
    checkAndReset(want: expectedC, got: bufC)
    printIfSmall(bufA, bufB, bufC, M, N, K)

    let matmulThreadgroupT = makePipeline(device, "matmul_threadgroup_t")
    benchmark(name: "matmul_threadgroup_t (16x16)") {
        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        cmdBuffer.label = "threadgroup_t"
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(matmulThreadgroupT)
        cmdEncoder.setBuffers([bufA, bufB, bufC], offsets: [0, 0, 0], range: 0..<3)
        cmdEncoder.setBytes(&M, length: MemoryLayout<Int>.stride, index: 3)
        cmdEncoder.setBytes(&N, length: MemoryLayout<Int>.stride, index: 4)
        cmdEncoder.setBytes(&K, length: MemoryLayout<Int>.stride, index: 5)
        cmdEncoder.dispatchThreads(MTLSizeMake(M, N, 1), threadsPerThreadgroup: MTLSizeMake(16, 16, 1))
        cmdEncoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
    }
    checkAndReset(want: expectedC, got: bufC)
    printIfSmall(bufA, bufB, bufC, M, N, K)

    let matmulSimdgroup = makePipeline(device, "matmul_simdgroup")
    benchmark(name: "matmul_simdgroup (8x8)") {
        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        cmdBuffer.label = "simdgroup"
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(matmulSimdgroup)
        cmdEncoder.setBuffers([bufA, bufB, bufC], offsets: [0, 0, 0], range: 0..<3)
        cmdEncoder.setBytes(&M, length: MemoryLayout<Int>.stride, index: 3)
        cmdEncoder.setBytes(&N, length: MemoryLayout<Int>.stride, index: 4)
        cmdEncoder.setBytes(&K, length: MemoryLayout<Int>.stride, index: 5)
        cmdEncoder.dispatchThreads(MTLSizeMake(M, N, 1), threadsPerThreadgroup: MTLSizeMake(8, 8, 1))
        cmdEncoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
    }
    checkAndReset(want: expectedC, got: bufC)
    printIfSmall(bufA, bufB, bufC, M, N, K)

    let matmulBaseline = makePipeline(device, "matmul_baseline")
    benchmark(name: "matmul_baseline (16x16)") {
        let cmdBuffer = cmdQueue.makeCommandBuffer()!
        cmdBuffer.label = "baseline"
        let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
        cmdEncoder.setComputePipelineState(matmulBaseline)
        cmdEncoder.setBuffers([bufA, bufB, bufC], offsets: [0, 0, 0], range: 0..<3)
        cmdEncoder.setBytes(&M, length: MemoryLayout<Int>.stride, index: 3)
        cmdEncoder.setBytes(&N, length: MemoryLayout<Int>.stride, index: 4)
        cmdEncoder.setBytes(&K, length: MemoryLayout<Int>.stride, index: 5)
        cmdEncoder.dispatchThreads(MTLSizeMake(M, N, 1), threadsPerThreadgroup: MTLSizeMake(16, 16, 1))
        cmdEncoder.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
    }
    checkAndReset(want: expectedC, got: bufC)
    printIfSmall(bufA, bufB, bufC, M, N, K)
}
