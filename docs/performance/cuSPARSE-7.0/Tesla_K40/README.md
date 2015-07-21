# Benchmarking
## Hardware
Tesla K40c

## Environment
OpenSUSE 13.2

cuSPARSE v7.0

Tesla driver 346.47

## Tool
[cusparse-bench](clSPARSE\src\benchmarks\cusparse-bench)

## Methodology
For each data point, we took 20 samples. Each sample consists of 20 calls
with a wait afterward. We benchmark with respect to the API, utilizing host timers
(not pure kernel time with ).  
Outlying samples beyond 1 standard deviation were removed.

Conversion routines benchmarked as number of Gi-Elements/s converted

SpM-dV routine calculated as Gi-Bytes/s
```c
( sizeof( cl_int )*( csrMtx.num_nonzeros + csrMtx.num_rows ) + sizeof( T ) * ( csrMtx.num_nonzeros + csrMtx.num_cols + csrMtx.num_rows ) ) / time_in_ns( );
```
