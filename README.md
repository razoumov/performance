## Problem description

We are solving the 2D heat diffusion problem

<!-- <img src="https://raw.githubusercontent.com/WestDRI/folio/refs/heads/main/fig/heatEquationInv.png" width="400"> -->
$$
\Huge
{\partial T(x,y,t)\over\partial t}={\partial^2 T\over\partial x^2}+{\partial^2 T\over\partial y^2}
$$

We discretize the solution $T(x,y,t)\approx T^{n}_{i,j}$ with $i=1,...,{\rm rows}$ and $j=1,...,{\rm cols}$, set the initial temperature at 25 degrees and apply the following boundary conditions:

- upper side $T^{n}_{0,1..{\rm cols}}\equiv 0$
- left side $T^{n}_{1..{\rm rows},0}\equiv 0$
- bottom side $T^{n}_{{\rm rows}+1,1..{\rm cols}} = {80\times j/\rm cols}$ (linearly increasing from 0 to 80 degrees)
- right side $T^{n}_{1..{\rm rows},{\rm cols}+1} = 80\times i/{\rm rows}$ (linearly increasing from 0 to 80 degrees)

We discretize the equation with forward Euler time stepping:

<!-- ![Forward Euler](https://folio.vastcloud.org/fig/discretizedHeatEquation1Inv.png) -->
$$
\Huge
{T^{n+1}_{i,j}-T^n_{i,j}\over\Delta t}=
{T^n_{i+1,j}-2T^n_{i,j}+T^n_{i-1,j}\over(\Delta x)^2}+
{T^n_{i,j+1}-2T^n_{i,j}+T^n_{i,j-1}\over(\Delta y)^2}
$$

and for simplicity we assume $\Delta x=\Delta y=1$ and $\Delta t=1/4$. Our finite difference equation then
becomes

<!-- <img src="https://folio.vastcloud.org/fig/discretizedHeatEquation2Inv.png" width="600"> -->
$$
\Huge
T^{n+1}_{i,j}={1\over 4}\left[T^n_{i+1,j}+T^n_{i-1,j}+T^n_{i,j+1}+T^n_{i,j-1}\right]
$$

i.e. at each time iteration we'll be computing the new temperature `Tnew` according to:

```chpl
Tnew[i,j] = 0.25 * (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1]);
```

## Implementations

- `original.c` is the original C+MPI distributed version
- `serial.chpl` is the serial version
- `singleLocale.chpl` is the single-locale version
- `block.chpl` is the BlockDist version
- `stencil.chpl` is the StencilDist version
- Slurm job submission scripts `runc.sh` and `runchapel.sh`

## Serial runs: C+MPI vs. single-local vs. multi-local Chapel

In this section and the next, I run on a small training cluster.

First, a serial C run:

```sh
>>> in original.c set rows=650, niter=10000
mpicc -O2 original.c -o original
srun --unbuffered --time=0:5:0 --mem-per-cpu=3600 ./original
At 650x650 after 7750 iterations reached delta=1.999846e-03 in 25.211740 seconds
```

Next I load single-locale Chapel 2.10.0 and run the code (multiple timings correspond to multiple runs):

```sh
chpl --fast serial.chpl
srun --unbuffered --time=0:5:0 --mem-per-cpu=3600 ./serial
At 650x650 after 7750 iterations reached delta=0.00199985 in 21.5249 21.4511 21.446 21.3584 21.6777 21.5501 seconds
```

The Chapel code runs slightly faster than C!

Next I load multi-locale Chapel 2.10.0 and compile and run the same serial code:

```sh
chpl --fast serial.chpl
srun --unbuffered --time=0:5:0 --mem-per-cpu=3600 ./serial -nl 1
At 650x650 after 7750 iterations reached delta=0.00199985 in 99.8576 99.6103 seconds
```

Even though I still have a serial code (all data local sitting on the same node, no `forall`), the code runs
~5X slower.

## Distributed-memory code on a single core

Still running on one CPU core, now I try a multi-locale code with a multi-locale Chapel build:

```sh
chpl --fast block.chpl
srun --unbuffered --time=1:0:0 --mem-per-cpu=3600 --nodes=1 --cpus-per-task=1 ./block -nl 1
At 650x650 after 7750 iterations reached delta=0.00199985 in 418.926 421.544 seconds
```

I see further slowdown, now running ~20X slower than the first Chapel code above, all on the same
hardware. This is the result of indexing into the distributed arrays, as discussed in [this
thread](https://chapel.discourse.group/t/distributed-code-performance/52244).

## Parallel C+MPI runs vs. distributed-memory Chapel

All runs in this section were performed on the InfiniBand cluster [Fir](https://docs.alliancecan.ca/wiki/Fir)
with `--mem=0 --nodes=5 --tasks-per-node=192`.

Multi-locale Chapel was built with `CHPL_LLVM=bundled CHPL_COMM=gasnet CHPL_COMM_SUBSTRATE=ibv
CHPL_LAUNCHER=gasnetrun_ibv GASNET_IBV_SPAWNER=ssh CHPL_TARGET_CPU=native CHPL_HWLOC=system
CHPL_TASKS=qthreads CHPL_TARGET_MEM=jemalloc`.

First I run the C+MPI implementation:

```sh
>>> in original.c set rows=50000, niter=10
mpicc -O2 original.c -o original50
>>> in original.c set rows=100000, niter=10
mpicc -O2 original.c -o original100
>>> in original.c set rows=200000, niter=10
mpicc -O2 original.c -o original200
sbatch runc.sh
```

and see the following timings:

```output
At 50000x50000 after 10 iterations reached delta=1.942879e+00 in 0.241712 seconds
At 100000x100000 after 10 iterations reached delta=1.942957e+00 in 0.941340 seconds
At 200000x200000 after 10 iterations reached delta=1.942996e+00 in 3.708167 seconds
```

Next I run the two multi-locale Chapel implementations:

```sh
# load multi-locale Chapel 2.10.0
module swap gcc gcc/13.3
module load libfabric/2.1.0
export GASNET_PHYSMEM_MAX=1G
CHDIR=/home/razoumov/c3/chapel-2.10.0
export PATH="$CHDIR/bin:$PATH"
alias printchplenv="$CHDIR/share/chapel/2.10/util/printchplenv --anonymize"

chpl --fast block.chpl
chpl --fast stencil.chpl
sbatch runchapel.sh
```

With BlockDist, I see the following timings:

```output
At 50000x50000 after 10 iterations reached delta=1.94288 in 2.86391 seconds
At 100000x100000 after 10 iterations reached delta=1.94296 in 10.4118 seconds
At 200000x200000 after 10 iterations reached delta=1.943 in 46.6852 seconds
```

With StencilDist, I see the following timings:

```output
At 50000x50000 after 10 iterations reached delta=1.94288 in 2.49755 seconds
At 100000x100000 after 10 iterations reached delta=1.94296 in 9.54453 seconds
At 200000x200000 after 10 iterations reached delta=1.943 in 44.005 seconds
```

### Removing the bottlenecks

As it turns out, the bottleneck is in the copy-back with large array slices `T[1..rows,1..cols] =
Tnew[1..rows,1..cols];`. There are two solutions:

First, we could simply use a whole-array assignment `T = Tnew` along with the modified boundary (which should
now be inserted into the main iteration loop).

With BlockDist this modification takes the code to within 2X of C+MPI performance, while having no real effect
with StencilDist:

| Resolution | BlockDist | StencilDist |
|------------|-------------|------------|
| 50,000^2 | 0.529417 seconds | 2.48717 seconds |
| 100,000^2 | 1.75888 | 9.52659 |
| 200,000^2 | 6.7203 | 43.6516 |

A better solution is to add a second loop:

```chpl
forall (i,j) in Dist do
  T[i,j] = Tnew[i,j];
```

With StencilDist we now come to within ~30% of C+MPI performance:

| Resolution | BlockDist | StencilDist |
|------------|-------------|------------|
| 50,000^2 | 0.558629 seconds | 0.322453 seconds |
| 100,000^2 | 1.92567 | 1.1081 |
| 200,000^2 | 7.34847 | 4.82358 |
