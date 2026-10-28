#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 650, cols = rows, niter = 10, nout = 1000;
  double tolerance = 2e-3;

  int dims[2] = {0, 0};             // MPI will decide on both in the next line
  MPI_Dims_create(size, 2, dims);   // create a balanced 2D processor grid
  int d1 = dims[0], d2 = dims[1];

  int periods[2] = {0, 0}; // non-periodic boundaries
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm); // create a new communicator

  // translate a rank into its corresponding grid coordinates
  int coords[2];
  MPI_Cart_coords(cart_comm, rank, 2, coords);
  int rx = coords[0], ry = coords[1];

  // find neighboring ranks (evaluate to MPI_PROC_NULL=-2 outside the grid);
  // passing MPI_PROC_NULL into any comms function treats it as a successful no-op
  int up, down, left, right;
  MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
  MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

  // compute local dimensions (distributing rows and cols across ranks)
  int localRows = rows / d1;
  int rowRem = rows % d1;
  int myRows = localRows + (rx < rowRem ? 1 : 0);
  int rowStart = rx * localRows + (rx < rowRem ? rx : rowRem) + 1; // 1-based global start
  int localCols = cols / d2;
  int colRem = cols % d2;
  int myCols = localCols + (ry < colRem ? 1 : 0);
  int colStart = ry * localCols + (ry < colRem ? ry : colRem) + 1; // 1-based global start
  //printf("rank %2d -> local indices %d..%d %d..%d\n", rank, rowStart, rowStart+myRows-1, colStart, colStart+myCols-1);

  // allocate local T and Tnew arrays with ghost cells (+2 for padding)
  double *T = (double *)calloc((myRows+2)*(myCols+2), sizeof(double));
  double *Tnew = (double *)calloc((myRows+2)*(myCols+2), sizeof(double));

#define IDX(i,j) ((i)*(myCols+2) + (j))   // preprocessor macro translating a local 2D coordinate (i,j) into a local 1D linear index

  for (int i = 1; i <= myRows; i++)
    for (int j = 1; j <= myCols; j++)
      T[IDX(i,j)] = 25.0;   // initial temperature

  if (ry == d2 - 1)   // if this rank owns the rightmost boundary
    for (int i = 1; i <= myRows; i++) {
      int iGlobal = rowStart + i - 1;
      T[IDX(i,myCols+1)] = 80.0 * iGlobal / rows;
    }
  if (rx == d1 - 1)   // if this rank owns the bottommost boundary
    for (int j = 1; j <= myCols; j++) {
      int jGlobal = colStart + j - 1;
      T[IDX(myRows+1,j)] = 80.0 * jGlobal / cols;
    }

  if (rank == 0)
    printf("Working with a matrix %ix%i to %i iterations or dT below %e\n", rows, cols, niter, tolerance);

  double delta = tolerance * 10.0;
  int count = 0;
  double start_time = MPI_Wtime();

  // buffers for ghost cell communication
  double *sendUp = (double *)malloc(myCols * sizeof(double));
  double *recvUp = (double *)malloc(myCols * sizeof(double));
  double *sendDown = (double *)malloc(myCols * sizeof(double));
  double *recvDown = (double *)malloc(myCols * sizeof(double));
  double *sendLeft = (double *)malloc(myRows * sizeof(double));
  double *recvLeft = (double *)malloc(myRows * sizeof(double));
  double *sendRight = (double *)malloc(myRows * sizeof(double));
  double *recvRight = (double *)malloc(myRows * sizeof(double));

  while (count < niter && delta >= tolerance) {
    count++;
    // 1. vertical halo exchange
    for (int j = 1; j <= myCols; j++) {
      sendUp[j-1] = T[IDX(1,j)];
      sendDown[j-1] = T[IDX(myRows,j)];
    }
    // info flows up: send to up and receive from down
    MPI_Sendrecv(sendUp, myCols, MPI_DOUBLE, up, 0, recvDown, myCols, MPI_DOUBLE, down, 0, cart_comm, MPI_STATUS_IGNORE);
    // info flows down: send to down and receive from up
    MPI_Sendrecv(sendDown, myCols, MPI_DOUBLE, down, 1, recvUp, myCols, MPI_DOUBLE, up, 1, cart_comm, MPI_STATUS_IGNORE);
    if (up != MPI_PROC_NULL)
      for (int j = 1; j <= myCols; j++)
	T[IDX(0,j)] = recvUp[j-1];
    if (down != MPI_PROC_NULL)
      for (int j = 1; j <= myCols; j++)
	T[IDX(myRows+1,j)] = recvDown[j-1];
    // 2. horizontal halo exchange
    for (int i = 1; i <= myRows; i++) {
      sendLeft[i-1] = T[IDX(i,1)];
      sendRight[i-1] = T[IDX(i,myCols)];
    }
    // info flows to the left: send to left, receive from right
    MPI_Sendrecv(sendLeft, myRows, MPI_DOUBLE, left, 2, recvRight, myRows, MPI_DOUBLE, right, 2, cart_comm, MPI_STATUS_IGNORE);
    // info flows to the right: send to right, receive from left
    MPI_Sendrecv(sendRight, myRows, MPI_DOUBLE, right, 3, recvLeft, myRows, MPI_DOUBLE, left, 3, cart_comm, MPI_STATUS_IGNORE);
    if (left != MPI_PROC_NULL)
      for (int i = 1; i <= myRows; i++)
	T[IDX(i,0)] = recvLeft[i-1];
    if (right != MPI_PROC_NULL)
      for (int i = 1; i <= myRows; i++)
	T[IDX(i,myCols+1)] = recvRight[i-1];
    // stencil computation
    double localDelta = 0.0;
    for (int i = 1; i <= myRows; i++) {
      for (int j = 1; j <= myCols; j++) {
	Tnew[IDX(i,j)] = 0.25 * (T[IDX(i-1,j)] + T[IDX(i+1,j)] + T[IDX(i,j-1)] + T[IDX(i,j+1)]);
	// Tnew[IDX(i,j)] = 0.99*T[IDX(i,j)];
	double diff = fabs(Tnew[IDX(i,j)] - T[IDX(i,j)]);
	if (diff > localDelta)
	  localDelta = diff;
      }
    }
    MPI_Allreduce(&localDelta, &delta, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    // update T = Tnew
    for (int i = 1; i <= myRows; i++)
      for (int j = 1; j <= myCols; j++)
	T[IDX(i,j)] = Tnew[IDX(i,j)];
    if (count % nout == 0 && rank == 0)
      printf("Iteration %i: delta=%f T=%f\n", count, delta, T[IDX(1,1)]);
  }
  double elapsed = MPI_Wtime() - start_time;
  if (rank == 0)
    printf("At %ix%i after %i iterations reached delta=%e in %f seconds\n", rows, cols, count, delta, elapsed);
  free(T); free(Tnew);
  free(sendUp); free(recvUp);
  free(sendDown); free(recvDown);
  free(sendLeft); free(recvLeft);
  free(sendRight); free(recvRight);
  MPI_Comm_free(&cart_comm);
  MPI_Finalize();
  return 0;
}
