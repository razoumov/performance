use Time;

config const rows = 650, cols = rows;
config const niter = 10, nout = 1000;
config const tolerance = 0.002: real;

var T, Tnew: [0..rows+1,0..cols+1] real;
T[1..rows,1..cols] = 25;                         // initial temperature

writef('Working with a matrix %ix%i to %i iterations or dT below %r\n', rows, cols, niter, tolerance);

for i in 1..rows do T[i,cols+1] = 80.0*i/rows;   // right-side boundary
for j in 1..cols do T[rows+1,j] = 80.0*j/cols;   // bottom-side boundary

writef('Iteration 0: T=%r\n', T[1,1]);
var count = 0: int;
var delta = tolerance*10;    // some safe initial large value
var watch: stopwatch;

watch.start();
while (count < niter && delta >= tolerance) {
  count += 1;    // update the iteration counter
  delta = 0;
  forall (i,j) in {1..rows,1..cols} with (max reduce delta) {
    Tnew[i,j] = 0.25 * (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1]);
    var tmp = abs(Tnew[i,j]-T[i,j]);
    if tmp > delta then delta = tmp;
  }
  T[1..rows,1..cols] = Tnew[1..rows,1..cols];
  if count%nout == 0 then writef('Iteration %i: delta=%r T=%r\n', count, delta, T[1,1]);
}
watch.stop();

writef('At %ix%i after %i iterations reached delta=%r in %r seconds\n', rows, cols, count, delta, watch.elapsed());
