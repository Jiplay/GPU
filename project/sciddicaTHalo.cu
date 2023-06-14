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
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value)                              \
  ((M)[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))] = (value))
#define BUF_GET(M, rows, columns, n, i, j)                                     \
  (M[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))])

// Halo cells parameters
#define MASK_WIDTH 3
#define TILE_WIDTH 14
#define TILE_BLOCK 16

// ----------------------------------------------------------------------------
// I/O functions and memory management
// ----------------------------------------------------------------------------
void readHeaderInfo(char *path, int &nrows, int &ncols, double &nodata) 
{
  FILE *f;

  if ((f = fopen(path, "r")) == 0) 
  {
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  // Read the header
  char str[STRLEN];
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  ncols = atoi(str); // ncols
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  nrows = atoi(str); // nrows
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str); // xllcorner
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str); // yllcorner
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str); // cellsize
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  nodata = atof(str); // NODATA_value
}

bool loadGrid2D(double *M, int rows, int columns, char *path) 
{
  FILE *f = fopen(path, "r");

  if (!f) 
  {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < columns; ++j) 
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
  for (int i = 0; i < rows; ++i) 
  {
    for (int j = 0; j < columns; ++j) 
    {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

double *addLayer2D(int rows, int columns) {
  double *tmp;

  cudaMallocManaged(&tmp, sizeof(double) * rows * columns);

  if (!tmp)
    return NULL;
  return tmp;
}

__global__ void sciddicaTSimulationInit(int r, int c, double *Sz, double *Sh) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i == 0 || j == 0 || (i >= r - 1) || (j >= c - 1))
    return;
  double z, h;
  h = GET(Sh, c, i, j);

  if (h > 0.0) 
  {
    z = GET(Sz, c, i, j);
    SET(Sz, c, i, j, z - h);
  }
}

__global__ void sciddicaTResetFlows(int r, int c, double *Sf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i == 0 || j == 0 || (i >= r - 1) || (j >= c - 1))
    return;
  BUF_SET(Sf, r, c, 0, i, j, 0.0);
  BUF_SET(Sf, r, c, 1, i, j, 0.0);
  BUF_SET(Sf, r, c, 2, i, j, 0.0);
  BUF_SET(Sf, r, c, 3, i, j, 0.0);
}
__global__ void sciddicaTFlowsComputation(int r, int c, double nodata, int *Xi, int *Xj,
                                                    double *Sz, double *Sh,
                                                    double *Sf, double p_r,
                                                    double p_epsilon) {
  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  __shared__ double Sz_ds[TILE_BLOCK][TILE_BLOCK];
  __shared__ double Sh_ds[TILE_BLOCK][TILE_BLOCK];

  unsigned int col_index = threadIdx.x + TILE_WIDTH * blockIdx.x;
  unsigned int row_index = threadIdx.y + TILE_WIDTH * blockIdx.y;
  unsigned int col_halo = col_index - MASK_WIDTH / 2;
  unsigned int row_halo = row_index - MASK_WIDTH / 2;

  if ((row_halo >= 0) && (row_halo < r) && (col_halo >= 0) && (col_halo < c)) 
  {
    Sz_ds[threadIdx.y][threadIdx.x] = GET(Sz, c, row_halo, col_halo);
    Sh_ds[threadIdx.y][threadIdx.x] = GET(Sh, c, row_halo, col_halo);
  } else 
  {
    Sz_ds[threadIdx.y][threadIdx.x] = 0.0;
    Sh_ds[threadIdx.y][threadIdx.x] = 0.0;
  }
  __syncthreads();

  if (row_index >= 1 && row_index < r - 1 && col_index >= 1 && col_index < c - 1) 
  {
    int index_i = threadIdx.y + MASK_WIDTH / 2;
    int index_j = threadIdx.x + MASK_WIDTH / 2;
    if (index_i >= 1 && index_i <= TILE_WIDTH && index_j >= 1 && index_j <= TILE_WIDTH) 
    {
      m = Sh_ds[index_i][index_j] - p_epsilon;
      u[0] = Sz_ds[index_i][index_j] + p_epsilon;

      z = Sz_ds[index_i + Xi[1]][index_j + Xj[1]];
      h = Sh_ds[index_i + Xi[1]][index_j + Xj[1]];
      u[1] = z + h;

      z = Sz_ds[index_i + Xi[2]][index_j + Xj[2]];
      h = Sh_ds[index_i + Xi[2]][index_j + Xj[2]];
      u[2] = z + h;

      z = Sz_ds[index_i + Xi[3]][index_j + Xj[3]];
      h = Sh_ds[index_i + Xi[3]][index_j + Xj[3]];
      u[3] = z + h;

      z = Sz_ds[index_i + Xi[4]][index_j + Xj[4]];
      h = Sh_ds[index_i + Xi[4]][index_j + Xj[4]];
      u[4] = z + h;

      do {
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
        {
          average /= cells_count;
        }

        for (n = 0; n < 5; n++) 
        {
          if ((average <= u[n]) && (!eliminated_cells[n])) 
          {
            eliminated_cells[n] = true;
            again = true;
          }
        }
      } while (again);

      if (!eliminated_cells[1]) 
      {
        BUF_SET(Sf, r, c, 0, row_index, col_index, (average - u[1]) * p_r);
      }
      if (!eliminated_cells[2]) 
      {
        BUF_SET(Sf, r, c, 1, row_index, col_index, (average - u[2]) * p_r);
      }
      if (!eliminated_cells[3]) 
      {
        BUF_SET(Sf, r, c, 2, row_index, col_index, (average - u[3]) * p_r);
      }
      if (!eliminated_cells[4]) 
      {
        BUF_SET(Sf, r, c, 3, row_index, col_index, (average - u[4]) * p_r);
      }
    }
  }
}

// This kernel benefits from a tiled implementation
__global__ void sciddicaTWidthUpdate(int r, int c, double nodata, int *Xi, int *Xj, double *Sz, double *Sh, double *Sf) 
{
  int row_index = threadIdx.y + TILE_WIDTH * blockIdx.y;
  int col_index = threadIdx.x + TILE_WIDTH * blockIdx.x;
  long row_halo = row_index - MASK_WIDTH / 2;
  long col_halo = col_index - MASK_WIDTH / 2;

  double h_next = 0.0;

  __shared__ double Sf_ds[TILE_BLOCK * ADJACENT_CELLS][TILE_BLOCK];

  if ((col_halo >= 0) && (col_halo < c) && (row_halo >= 0) && (row_halo < r)) 
  {
    Sf_ds[threadIdx.y][threadIdx.x] = BUF_GET(Sf, r, c, 0, row_halo, col_halo);
    Sf_ds[threadIdx.y + TILE_BLOCK][threadIdx.x] = BUF_GET(Sf, r, c, 1, row_halo, col_halo);
    Sf_ds[threadIdx.y + 2 * TILE_BLOCK][threadIdx.x] = BUF_GET(Sf, r, c, 2, row_halo, col_halo);
    Sf_ds[threadIdx.y + 3 * TILE_BLOCK][threadIdx.x] = BUF_GET(Sf, r, c, 3, row_halo, col_halo);
  } else 
  {
    Sf_ds[threadIdx.y][threadIdx.x] = 0.0;
  }
  __syncthreads();

  if (row_index >= 1 && col_index >= 1 && row_index < r - 1 && col_index < c - 1) 
  {
    int index_i = threadIdx.y + MASK_WIDTH / 2;
    int index_j = threadIdx.x + MASK_WIDTH / 2;
    if (index_i >= 1 && index_i <= TILE_WIDTH && index_j >= 1 && index_j <= TILE_WIDTH) 
    {
      h_next = GET(Sh, c, row_index, col_index);
      h_next += Sf_ds[index_i + Xi[1] + (TILE_BLOCK * 3)][index_j + Xj[1]] - Sf_ds[index_i][index_j];
      h_next += Sf_ds[index_i + Xi[2] + (TILE_BLOCK * 2)][index_j + Xj[2]] - Sf_ds[index_i + TILE_BLOCK][index_j];
      h_next += Sf_ds[index_i + Xi[3] + TILE_BLOCK][index_j + Xj[3]] - Sf_ds[index_i + (TILE_BLOCK * 2)][index_j];
      h_next += Sf_ds[index_i + Xi[4]][index_j + Xj[4]] - Sf_ds[index_i + (TILE_BLOCK * 3)][index_j];

      SET(Sh, c, row_index, col_index, h_next);
    }
  }
}
// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows; // r: grid rows
  int c = cols; // c: grid columns
  double *Sz;   // Sz: substate (grid) containing cells' altitude a.s.l.
  double *Sh;   // Sh: substate (grid) containing cells' flow thickness
  double *Sf;   // Sf: 4 substates containing the flows towards the 4 neighbors
  int *Xi;      // Xj: von Neuman neighborhood row coordinates (see below)
  int *Xj;      // Xj: von Neuman neighborhood col coordinates (see below)
  double p_r = P_R; // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;     // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); // steps: simulation steps

  Sz = addLayer2D(r, c); // Allocates the Sz substate grid
  Sh = addLayer2D(r, c); // Allocates the Sh substate grid
  Sf = addLayer2D(ADJACENT_CELLS * r, c); // Allocates the Sf substates grid, having one layer for
                      // each adjacent cell
  cudaMallocManaged(&Xi, sizeof(int) * 5);
  Xi[0] = 0;
  Xi[1] = -1;
  Xi[2] = 0;
  Xi[3] = 0;
  Xi[4] = 1;
  cudaMallocManaged(&Xj, sizeof(int) * 5);
  Xj[0] = 0;
  Xj[1] = 0;
  Xj[2] = -1;
  Xj[3] = 1;
  Xj[4] = 0;

  // printf("Loading data from file...\n");
  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);    // Load Sz from file
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]); // Load Sh from file

  int n = rows * cols;
  
  dim3 t_block_size(TILE_BLOCK , TILE_BLOCK, 1); 
  dim3 t_grid_size(ceil(rows / TILE_WIDTH), ceil(cols / TILE_WIDTH), 1);
  printf(">>>>Block size %d, Grid size: %d x %d\n",TILE_WIDTH, t_grid_size.x, t_grid_size.y);
  
  //setting up the grid and block sizes
  int THREADS_N = TILE_BLOCK;
  dim3 block_size(THREADS_N, THREADS_N, 1);
  dim3 grid_size(ceil(sqrt(n / (THREADS_N * THREADS_N))), ceil(sqrt(n / (THREADS_N * THREADS_N))), 1);
  printf(">>>>Block size %d, Grid size: %d x %d\n",THREADS_N, grid_size.x, grid_size.y);

  //initializing simulation
  sciddicaTSimulationInit<<<grid_size, block_size>>>(r, c, Sz, Sh);

  cudaDeviceSynchronize();
  util::Timer cl_timer;

  //main loop with kernel calls for each step
  for (int s = 0; s < steps; ++s) {
    sciddicaTResetFlows<<<grid_size, block_size>>>(r, c, Sf);
    cudaDeviceSynchronize();
    
    sciddicaTFlowsComputation<<<t_grid_size, t_block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon);
    cudaDeviceSynchronize();

    sciddicaTWidthUpdate<<<t_grid_size, t_block_size>>>(r, c, nodata, Xi, Xj, Sz,Sh, Sf);
    cudaDeviceSynchronize();
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);

  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);

  printf("Releasing memory...\n");

  cudaFree(Sz);
  cudaFree(Sh);
  cudaFree(Sf);
  cudaFree(Xi);
  cudaFree(Xj);
  return 0;
}