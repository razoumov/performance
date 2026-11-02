use StencilDist, IO;

config const rows = 650, cols = rows;
config const niter = 10, nout = 1000;
config const tolerance = 0.002: real;

const mesh = {1..rows, 1..cols};
const Dist = mesh dmapped new stencilDist(mesh, fluff=(1,1), periodic=false);
var T, Tnew: [Dist] real;
T[1..rows,1..cols] = 25;   // initial temperature

writef('Working with a matrix %ix%i to %i iterations or dT below %r\n', rows, cols, niter, tolerance);

for i in 1..rows {
  T[i,cols+1] = 80.0*i/rows;   // right-side boundary
  Tnew[i,cols+1] = T[i,cols+1];
}
for j in 1..cols {
  T[rows+1,j] = 80.0*j/cols;   // bottom-side boundary
  Tnew[rows+1,j] = T[rows+1,j];
}

writef('Iteration 0: T=%r\n', T[1,1]);
var count = 0: int;
var delta = tolerance*10.0;   // some safe initial large value

while (count < niter && delta >= tolerance) {
  count += 1;
  delta = 0;
  T.updateFluff(); // copy the updated ghost cell values from adjacent locales
  forall (i,j) in Dist do
    Tnew[i,j] = 0.25 * (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1]);
  delta = max reduce abs(Tnew - T);
  const Temp = T;  T = Tnew;  Tnew = Temp;
  if count%nout == 0 then writef('Iteration %i: delta=%r T=%r\n', count, delta, T[1,1]);
}
