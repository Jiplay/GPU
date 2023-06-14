#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util.hpp"

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define P_R 0.5
#define P_EPSILON 0.001
#define ADJACENT_CELLS 4
#define STRLEN 256

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char* path, int &nrows, int &ncols, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata)
{
  FILE* f;
  
  if ( (f = fopen(path,"r") ) == 0){
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  //Reading the header
  char str[STRLEN];
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); ncols = atoi(str);      //ncols
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nrows = atoi(str);      //nrows
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //xllcorner = atof(str);  //xllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //yllcorner = atof(str);  //yllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //cellsize = atof(str);   //cellsize
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);     //NODATA_value 
}

bool loadGrid2D(double *M, int rows, int columns, char *path)
{
  FILE *f = fopen(path, "r");

  if (!f) {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < columns; j++)
    {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }

  fclose(f);

  return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

double* addLayer2D(int rows, int columns)
{
  double *tmp;
  
  cudaMallocManaged(&tmp, sizeof(double) * rows * columns);
  if (!tmp)
    return NULL;
  return tmp;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
__global__ void sciddicaTSimulationInitKernel(int r, int c, double *Sz,
                                              double *Sh)
{
  int row_index = threadIdx.y + blockDim.y * blockIdx.y;
  int col_index = threadIdx.x + blockDim.x * blockIdx.x;
  int row_stride = blockDim.y * gridDim.y;
  int col_stride = blockDim.x * gridDim.x;

  for (int row = row_index + 1; row < r - 1; row += row_stride)
  {
    for (int col = col_index + 1; col < c - 1; col += col_stride)
    {
      double z, h;
      h = GET(Sh, c, row, col);

      if (h > 0.0)
      {
        z = GET(Sz, c, row, col);
        SET(Sz, c, row, col, z - h);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
__global__ void sciddicaTResetFlowsKernel(int r, int c, double nodata, double *Sf)
{
  int row_index = threadIdx.y + blockDim.y * blockIdx.y;
  int col_index = threadIdx.x + blockDim.x * blockIdx.x;
  int row_stride = blockDim.y * gridDim.y;
  int col_stride = blockDim.x * gridDim.x;

  for (int row = row_index + 1; row < r - 1; row += row_stride)
  {
    for (int col = col_index + 1; col < c - 1; col += col_stride)
    {
      BUF_SET(Sf, r, c, 0, row, col, 0.0);
      BUF_SET(Sf, r, c, 1, row, col, 0.0);
      BUF_SET(Sf, r, c, 2, row, col, 0.0);
      BUF_SET(Sf, r, c, 3, row, col, 0.0);
    }
  }
}

__global__ void sciddicaTFlowsComputationKernel(int r, int c, double nodata, int *Xi, int *Xj, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon)
{
  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  int row_index = threadIdx.y + blockDim.y * blockIdx.y;
  int col_index = threadIdx.x + blockDim.x * blockIdx.x;
  int row_stride = blockDim.y * gridDim.y;
  int col_stride = blockDim.x * gridDim.x;

  for (int row = row_index + 1; row < r - 1; row += row_stride)
  {
    for (int col = col_index + 1; col < c - 1; col += col_stride)
    {
      m = GET(Sh, c, row, col) - p_epsilon;
      u[0] = GET(Sz, c, row, col) + p_epsilon;

      u[1] = z + h;                                         
      z = GET(Sz, c, row + Xi[1], col + Xj[1]);
      h = GET(Sh, c, row + Xi[1], col + Xj[1]);

      u[2] = z + h;                                         
      z = GET(Sz, c, row + Xi[2], col + Xj[2]);
      h = GET(Sh, c, row + Xi[2], col + Xj[2]);

      u[3] = z + h;
      z = GET(Sz, c, row + Xi[3], col + Xj[3]);
      h = GET(Sh, c, row + Xi[3], col + Xj[3]);

      u[4] = z + h;
      z = GET(Sz, c, row + Xi[4], col + Xj[4]);
      h = GET(Sh, c, row + Xi[4], col + Xj[4]);

      do
      {
        again = false;
        average = m;
        cells_count = 0;

        for (n = 0; n < 5; n++)
          if (!eliminated_cells[n])
          {
            average += u[n];
            cells_count++;
          }

        if (cells_count != 0)
          average /= cells_count;

        for (n = 0; n < 5; n++)
          if ((average <= u[n]) && (!eliminated_cells[n]))
          {
            eliminated_cells[n] = true;
            again = true;
          }
      } while (again);

      if (!eliminated_cells[1]) BUF_SET(Sf, r, c, 0, row, col, (average - u[1]) * p_r);
      if (!eliminated_cells[2]) BUF_SET(Sf, r, c, 1, row, col, (average - u[2]) * p_r);
      if (!eliminated_cells[3]) BUF_SET(Sf, r, c, 2, row, col, (average - u[3]) * p_r);
      if (!eliminated_cells[4]) BUF_SET(Sf, r, c, 3, row, col, (average - u[4]) * p_r);
    }
  }
}

__global__ void sciddicaTWidthUpdateKernel(int r, int c, double nodata, int *Xi, int *Xj, double *Sz, double *Sh, double *Sf)
{
  int row_index = threadIdx.y + blockDim.y * blockIdx.y;
  int col_index = threadIdx.x + blockDim.x * blockIdx.x;
  int row_stride = blockDim.y * gridDim.y;
  int col_stride = blockDim.x * gridDim.x;

  for (int row = row_index + 1; row < r - 1; row += row_stride)
  {
    for (int col = col_index + 1; col < c - 1; col += col_stride)
    {
      double h_next;
      h_next = GET(Sh, c, row, col);
      h_next += BUF_GET(Sf, r, c, 3, row + Xi[1], col + Xj[1]) - BUF_GET(Sf, r, c, 0, row, col);
      h_next += BUF_GET(Sf, r, c, 2, row + Xi[2], col + Xj[2]) - BUF_GET(Sf, r, c, 1, row, col);
      h_next += BUF_GET(Sf, r, c, 1, row + Xi[3], col + Xj[3]) - BUF_GET(Sf, r, c, 2, row, col);
      h_next += BUF_GET(Sf, r, c, 0, row + Xi[4], col + Xj[4]) - BUF_GET(Sf, r, c, 3, row, col);

      SET(Sh, c, row, col, h_next);
    }
  }
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows; // r: grid rows
  int c = cols; // c: grid columns
  double *Sz;   // Sz: substate (grid) containing the cells' altitude a.s.l.
  double *Sh;   // Sh: substate (grid) containing the cells' flow thickness
  double *Sf;   // Sf: 4 substates containing the flows towards the 4 neighs

  int *Xi;
  int *Xj;

  cudaMallocManaged(&Xi, sizeof(int) * 5);
  cudaMallocManaged(&Xj, sizeof(int) * 5);

  Xi[0] = 0;
  Xi[1] = -1;
  Xi[2] = 0;
  Xi[3] = 0;
  Xi[4] = 1;
  
  Xj[0] = 0;
  Xj[1] = 0;
  Xj[2] = -1;
  Xj[3] = 1;
  Xj[4] = 0;

  double p_r = P_R;                 // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;     // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); // steps: simulation steps

  int dim_x = 32;
  int dim_y = 32;
  dim3 block_size(dim_x, dim_y, 1);
  dim3 grid_size(ceil(rows / dim_y), ceil(cols / dim_x), 1);

  int comp_dim_x = 16;
  int comp_dim_y = 16;
  dim3 comp_block_size(comp_dim_y, comp_dim_x, 1);
  dim3 comp_grid_size(ceil(rows / comp_dim_y), ceil(cols / comp_dim_x), 1);

  Sz = addLayer2D(r, c); // Allocates the Sz substate grid
  Sh = addLayer2D(r, c); // Allocates the Sh substate grid
  Sf = addLayer2D(ADJACENT_CELLS * r, c); // Allocates the Sf substates grid,
                      //   having one layer for each adjacent cell

  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);    // Load Sz from file
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]); // Load Sh from file

  // Apply the init kernel (elementary process) to the whole domain grid
  // (cellular space)
  sciddicaTSimulationInitKernel<<<grid_size, block_size>>>(r, c, Sz, Sh);

  util::Timer cl_timer;
  // simulation loop with kernel applications
  for (int s = 0; s < steps; ++s)
  {
    sciddicaTResetFlowsKernel<<<grid_size, block_size>>>(r, c, nodata, Sf);

    sciddicaTFlowsComputationKernel<<<comp_grid_size, comp_block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon);

    sciddicaTWidthUpdateKernel<<<grid_size, block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf);
  }
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);

  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]); // Save Sh to file

  printf("Releasing memory...\n");
  cudaFree(Sz);
  cudaFree(Sh);
  cudaFree(Sf);

  return 0;
}