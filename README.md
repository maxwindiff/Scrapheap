### Matmul

Current performance on an M1 Pro:

```
                                     mps: 0.0054041541 seconds
         matmul_2d_threadtile (8x8, 4x4): 0.0147047835 seconds
         matmul_1d_threadtile (8x8, 1x4): 0.0193941249 seconds
              matmul_threadgroup (16x16): 0.0243510541 seconds
            matmul_threadgroup_t (16x16): 0.0366692125 seconds
                  matmul_simdgroup (8x8): 0.0331840248 seconds
                 matmul_baseline (16x16): 0.0544398832 seconds
```

### Reduce

Trying out the algorithm from
https://kieber-emmons.medium.com/optimizing-parallel-reduction-in-metal-for-apple-m1-8e8677b49b01

