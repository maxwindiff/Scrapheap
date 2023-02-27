import Foundation
import Metal
import MetalPerformanceShaders

func makeFilledDeviceVec<T>(_ device:MTLDevice, _ N:Int, _ value:T) -> MTLBuffer {
    let buf = device.makeBuffer(length: MemoryLayout<T>.stride * N)!
    let contents = buf.contents().bindMemory(to: T.self, capacity: N)
    for i in 0..<N {
        contents[i] = value
    }
    return buf
}

func makeRandomDeviceMtx(_ device:MTLDevice, _ M:Int, _ N:Int) -> (MTLBuffer, MPSMatrix) {
    let buf = device.makeBuffer(length: MemoryLayout<Float>.stride * M * N)!
    let desc = MPSMatrixDescriptor(rows: M, columns: N, rowBytes: MemoryLayout<Float>.stride * N, dataType: .float32)
    let mtx = MPSMatrix(buffer: buf, descriptor: desc)
    let rnd = MPSMatrixRandomMTGP32(device: device,
                                    destinationDataType: .float32,
                                    seed: 0,
                                    distributionDescriptor: .uniformDistributionDescriptor(withMinimum: 0, maximum: 1))
    let cmdQueue = device.makeCommandQueue()!
    let cmdBuffer = cmdQueue.makeCommandBuffer()!
    cmdBuffer.label = "MPSMatrixRandom"
    rnd.encode(commandBuffer: cmdBuffer, destinationMatrix: mtx)
    cmdBuffer.commit()
    cmdBuffer.waitUntilCompleted()
    return (buf, mtx)
}

func clearBuffer(_ buf:MTLBuffer) {
    memset(buf.contents(), 0, buf.length)
}

func printMtx(_ heading:String, _ mtx:UnsafePointer<Float>, _ rows:Int, _ cols:Int) {
    print(heading)
    for i in 0..<rows {
        print(i == 0 ? "[" : " ", terminator: "")
        for j in 0..<cols {
            print(String(format: " %9.6f", mtx[i * cols + j]), terminator: "")
        }
        print(i == rows-1 ? " ];" : " ;")
    }
    print("")
}

func printDeviceMtx(_ heading:String, _ buf:MTLBuffer, _ rows:Int, _ cols:Int) {
    let contents = buf.contents().bindMemory(to: Float.self, capacity: rows * cols)
    printMtx(heading, contents, rows, cols)
}

func printVec(_ heading:String, _ mtx:UnsafePointer<UInt32>, _ len:Int) {
    print(heading, "[", terminator: "")
    for j in 0..<len {
        print(String(format: " %9d", mtx[j]), terminator: "")
    }
    print(" ];")
}

func printDeviceVec(_ heading:String, _ buf:MTLBuffer, _ len:Int) {
    let contents = buf.contents().bindMemory(to: UInt32.self, capacity: len)
    printVec(heading, contents, len)
}

func printIfSmall(_ A:MTLBuffer, _ B:MTLBuffer, _ C:MTLBuffer, _ M:Int, _ N:Int, _ K:Int) {
    if M <= 16 {
        printDeviceMtx("A=", A, M, K)
        printDeviceMtx("B=", B, K, N)
        printDeviceMtx("C=", C, M, N)
    }
}

func equals(_ bufA:MTLBuffer, _ bufB:MTLBuffer) -> Bool {
    let len = bufA.length / MemoryLayout<Float>.stride
    let a = bufA.contents().bindMemory(to: Float.self, capacity: len)
    let b = bufB.contents().bindMemory(to: Float.self, capacity: len)
    var maxDiff:Float = 0.0;
    for i in 0..<len {
        maxDiff = max(maxDiff, abs(a[i] - b[i]));
    }
    return maxDiff < 1e-8
}

func checkAndReset(want:MTLBuffer, got:MTLBuffer) {
    if !equals(want, got) {
        print("!! NOT EQUAL !!")
    }
    clearBuffer(got)
}

func makePipeline(_ device: MTLDevice, _ name: String, _ constants:MTLFunctionConstantValues = MTLFunctionConstantValues()) -> MTLComputePipelineState {
    let library = device.makeDefaultLibrary()!
    let desc = MTLComputePipelineDescriptor()
    desc.computeFunction = try! library.makeFunction(name: name, constantValues: constants)
    desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
    let (pso, _) = try! device.makeComputePipelineState(descriptor: desc, options: MTLPipelineOption())

    let archiveDesc = MTLBinaryArchiveDescriptor()
    let archive = try! device.makeBinaryArchive(descriptor: archiveDesc)
    try! archive.addComputePipelineFunctions(descriptor: desc)
    try! archive.serialize(to: NSURL.fileURL(withPath: "/tmp/" + name + ".metallib"))

    return pso
}

func measure(_ closure: () -> Void) -> Duration {
    closure(); // warm up

    let clock = ContinuousClock()
    let trials = 10;
    var totalTime: Duration = .zero
    for _ in 0..<trials {
        totalTime += clock.measure(closure)
    }
    return totalTime / trials;
}

func benchmark(name: String, _ closure: () -> Void) {
    let duration = measure(closure)
    print(String(repeating: " ", count: 40 - name.count), terminator: "")
    print("\(name): \(duration)")
}
