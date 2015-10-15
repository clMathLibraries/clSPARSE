# Benchmarking
## Hardware
w9100

## Environment
Ubuntu 14.04

clSPARSE v0.8.0.0

[Catalyst FirePro](http://support.amd.com/en-us/download/workstation?os=Linux%20x86_64#catalyst-pro) 14.502.1040

## Tool
[clsparse-bench](clSPARSE\src\benchmarks\clsparse-bench)

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
SpGEMM routine calculated as Mega-Flops/s
```c
 (2 * (upper bound of number of nonzeros of result matrix))/ time_in_ms( ) ;
 ```
 