import Foundation
import Metal
import MetalPerformanceShaders

func reduceOnce(_ cmdQueue: MTLCommandQueue, _ pso: MTLComputePipelineState, _ bufIn: MTLBuffer, _ bufOut: MTLBuffer,
                _ N: Int, _ BLOCK_SIZE: Int, _ GRAIN_SIZE: Int) {
    let cmdBuffer = cmdQueue.makeCommandBuffer()!
    cmdBuffer.label = "reduce"
    let cmdEncoder = cmdBuffer.makeComputeCommandEncoder()!
    cmdEncoder.setComputePipelineState(pso)

    var n = N;
    cmdEncoder.setBuffer(bufOut, offset: 0, index: 0)
    cmdEncoder.setBuffer(bufIn, offset: 0, index: 1)
    cmdEncoder.setBytes(&n, length: MemoryLayout<Int>.stride, index: 2)
    cmdEncoder.setThreadgroupMemoryLength(MemoryLayout<Float>.stride * BLOCK_SIZE, index: 0)
    cmdEncoder.dispatchThreads(MTLSizeMake(N/GRAIN_SIZE, 1, 1), threadsPerThreadgroup: MTLSizeMake(BLOCK_SIZE, 1, 1))
    cmdEncoder.endEncoding()
    cmdBuffer.commit()
    cmdBuffer.waitUntilCompleted()
}

func testReduce() {
    let BLOCK_SIZE = 256, GRAIN_SIZE = 4, FACTOR = BLOCK_SIZE * GRAIN_SIZE, N = 8192 * 8192

    let device = MTLCreateSystemDefaultDevice()!
    let cmdQueue = device.makeCommandQueue()!

    let bufIn = makeFilledDeviceVec(device, N, UInt32(1))
    let buf1 = device.makeBuffer(length: MemoryLayout<UInt32>.stride * (N / FACTOR))!
    let buf2 = device.makeBuffer(length: MemoryLayout<UInt32>.stride * (N / FACTOR / FACTOR))!
    clearBuffer(buf1)
    clearBuffer(buf2)

    let contents = bufIn.contents().bindMemory(to: UInt32.self, capacity: N)
    var sum:UInt32 = 0
    for i in 0..<N {
        sum += contents[i]
    }
    print("expected_sum =", sum)

    let constants = MTLFunctionConstantValues()
    var localAlgorithm = 0;
    var globalAlgorithm = 0;
    var disableBoundCheck = true;
    var useShuffle = true;
    constants.setConstantValue(&localAlgorithm, type: .int, index: 0)
    constants.setConstantValue(&globalAlgorithm, type: .int, index: 1)
    constants.setConstantValue(&disableBoundCheck, type: .bool, index: 2)
    constants.setConstantValue(&useShuffle, type: .bool, index: 3)
    let reduceMKE = makePipeline(device, "reduce_sum_uint32_256threads_4way", constants)

    benchmark(name: "reduce_sum_uint32_256threads_4way") {
        reduceOnce(cmdQueue, reduceMKE, bufIn, buf1, N, BLOCK_SIZE, GRAIN_SIZE)
        reduceOnce(cmdQueue, reduceMKE, buf1, buf2, N / FACTOR, BLOCK_SIZE, GRAIN_SIZE)
        // TODO: not sure why, but doing it one more round gives incorrect results
    }
    printDeviceVec("buf2 =", buf2, N / FACTOR / FACTOR)
}
