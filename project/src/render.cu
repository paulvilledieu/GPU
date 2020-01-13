#include "render.hpp"
#include <cassert>
#include <iostream>

  [[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  //spdlog::error("{} ({}, line: {})", msg, fname, line);
  //spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

// Executed on device, called by host
__global__ void kernel_dilation(unsigned char* dst_device, unsigned char *src_device, int structuring_radius,
    int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
  int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
  int end_x = x+structuring_radius > width - 1 ? width - 1 : x+structuring_radius;
  int end_y = y+structuring_radius > height - 1 ? height - 1 : y+structuring_radius;
  bool stop = false;

  for (int i = start_y; i <= end_y && !stop; ++i)
  {
    for (int j = start_x; j <= end_x && !stop; ++j)
    {
      if (src_device[i*width+ j] == 0)
      {
        dst_device[y*width+ x] = 0;
        stop = true;
      }
    }
  }

  if (!stop)
    dst_device[y*width+x] = 255;
}

__global__ void kernel_erosion(unsigned char* dst_device, unsigned char *src_device, int structuring_radius,
    int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
  int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
  int end_x = x+structuring_radius > width - 1 ? width - 1 : x+structuring_radius;
  int end_y = y+structuring_radius > height - 1 ? height - 1 : y+structuring_radius;
  bool stop = false;

  for (int i = start_y; i <= end_y && !stop; ++i)
  {
    for (int j = start_x; j <= end_x && !stop; ++j)
    {
      if (src_device[i*width+ j] == 255)
      {
        dst_device[y*width+ x] = 255;
        stop = true;
      }
    }
  }

  if (!stop)
    dst_device[y*width+x] = 0;
}

__global__ void kernel_dilation_sep_1(unsigned char* tmp_device, unsigned char *src_device,
    int structuring_radius, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
  int end_x = x+structuring_radius > width - 1? width - 1: x+structuring_radius;
  bool stop = false;

  for (int i = start_x; i <= end_x && !stop; ++i)
  {
    if (src_device[y*width+ i] == 0)
    {
      tmp_device[y*width+ x] = 0;
      stop = true;
    }
  }

  if (!stop)
    tmp_device[y*width+x] = 255;
}

__global__ void kernel_dilation_sep_2(unsigned char* dst_device, unsigned char *tmp_device,
    int structuring_radius, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
  int end_y = y+structuring_radius > height - 1? height - 1: y+structuring_radius;
  bool stop = false;

  for (int j = start_y; j <= end_y && !stop; ++j)
  {
    if (tmp_device[j*width+ x] == 0)
    {
      dst_device[y*width+ x] = 0;
      stop = true;
    }
  }

  if (!stop)
    dst_device[y*width+x] = 255;
}

__global__ void kernel_erosion_sep_1(unsigned char* tmp_device, unsigned char *src_device,
    int structuring_radius, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
  int end_x = x+structuring_radius > width - 1? width - 1: x+structuring_radius;
  bool stop = false;

  for (int i = start_x; i <= end_x && !stop; ++i)
  {
    if (src_device[y*width+ i] == 255)
    {
      tmp_device[y*width+ x] = 255;
      stop = true;
    }
  }

  if (!stop)
    tmp_device[y*width+x] = 0;
}

__global__ void kernel_erosion_sep_2(unsigned char* dst_device, unsigned char *tmp_device,
    int structuring_radius, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
  int end_y = y+structuring_radius > height - 1? height - 1: y+structuring_radius;
  bool stop = false;

  for (int j = start_y; j <= end_y && !stop; ++j)
  {
    if (tmp_device[j*width+ x] == 255)
    {
      dst_device[y*width+ x] = 255;
      stop = true;
    }
  }

  if (!stop)
    dst_device[y*width+x] = 0;
}

__global__ void kernel_dilation_sep_1_shared(unsigned char* tmp_device, unsigned char *src_device,
    int structuring_radius, int width, int height, int tile_width, int tile_height)

{
  int x = blockIdx.x * tile_width + threadIdx.x;
  int y = blockIdx.y * tile_height + threadIdx.y;

  if (x >= width || y >= height)
    return;

  unsigned char *block_ptr = src_device + blockIdx.x;
  extern __shared__ unsigned char tile[];

  for (int j = threadIdx.y; j < tile_height; j += blockDim.y)
  {
    if (j >= height)
      continue;
    tile[j] = block_ptr[j*width];
  }

  __syncthreads();
  for (int i = threadIdx.y; i < tile_height; i += blockDim.y)
  {
    int start = i - structuring_radius;
    bool stop = false;

    for (int j = start; j <= start + structuring_radius * 2 && !stop; ++j)
    {
      if (j < 0)
        continue;
      if (j < height && tile[j] == 0)
      {
        stop = true;
        tmp_device[i*width+blockIdx.x] = 0;
      }
    }
    if (!stop)
      tmp_device[i*width+blockIdx.x] = 255;
  }
}

__global__ void kernel_dilation_sep_2_shared(unsigned char* tmp_device, unsigned char *src_device,
    int structuring_radius, int width, int height, int tile_width, int tile_height)

{
  int x = blockIdx.x * tile_width + threadIdx.x;
  int y = blockIdx.y * tile_height + threadIdx.y;

  if (x >= width || y >= height)
    return;

  unsigned char *block_ptr = src_device + blockIdx.y*width;
  extern __shared__ unsigned char tile[];

  for (int j = threadIdx.x; j < tile_width; j += blockDim.x)
  {
    if (j >= width)
      continue;
    tile[j] = block_ptr[j];
  }

  __syncthreads();
  for (int i = threadIdx.x; i < tile_width; i += blockDim.x)
  {
    int start = i - structuring_radius;
    bool stop = false;

    for (int j = start; j <= start + structuring_radius * 2 && !stop; ++j)
    {
      if (j < 0)
        continue;
      if (j < width && tile[j] == 0)
      {
        stop = true;
        tmp_device[blockIdx.y*width+i] = 0;
      }
    }
    if (!stop)
      tmp_device[blockIdx.y*width+i] = 255;
  }
}

__global__ void kernel_erosion_sep_1_shared(unsigned char* tmp_device, unsigned char *src_device,
    int structuring_radius, int width, int height, int tile_width, int tile_height)

{
  int x = blockIdx.x * tile_width + threadIdx.x;
  int y = blockIdx.y * tile_height + threadIdx.y;

  if (x >= width || y >= height)
    return;

  unsigned char *block_ptr = src_device + blockIdx.x;
  extern __shared__ unsigned char tile[];

  for (int j = threadIdx.y; j < tile_height; j += blockDim.y)
  {
    if (j >= height)
      continue;
    tile[j] = block_ptr[j*width];
  }

  __syncthreads();
  for (int i = threadIdx.y; i < tile_height; i += blockDim.y)
  {
    int start = i - structuring_radius;
    bool stop = false;

    for (int j = start; j <= start + structuring_radius * 2 && !stop; ++j)
    {
      if (j < 0)
        continue;
      if (j < height && tile[j] == 0)
      {
        stop = true;
        tmp_device[i*width+blockIdx.x] = 255;
      }
    }
    if (!stop)
      tmp_device[i*width+blockIdx.x] = 0;
  }
}

__global__ void kernel_erosion_sep_2_shared(unsigned char* tmp_device, unsigned char *src_device,
    int structuring_radius, int width, int height, int tile_width, int tile_height)

{
  int x = blockIdx.x * tile_width + threadIdx.x;
  int y = blockIdx.y * tile_height + threadIdx.y;

  if (x >= width || y >= height)
    return;

  unsigned char *block_ptr = src_device + blockIdx.y*width;
  extern __shared__ unsigned char tile[];

  for (int j = threadIdx.x; j < tile_width; j += blockDim.x)
  {
    if (j >= width)
      continue;
    tile[j] = block_ptr[j];
  }

  __syncthreads();
  for (int i = threadIdx.x; i < tile_width; i += blockDim.x)
  {
    int start = i - structuring_radius;
    bool stop = false;

    for (int j = start; j <= start + structuring_radius * 2 && !stop; ++j)
    {
      if (j < 0)
        continue;
      if (j < width && tile[j] == 0)
      {
        stop = true;
        tmp_device[blockIdx.y*width+i] = 255;
      }
    }
    if (!stop)
      tmp_device[blockIdx.y*width+i] = 0;
  }
}


void run_kernel(std::string type, unsigned char* dst_device, unsigned char* src_device,
    int width, int height, int structuring_radius, std::ptrdiff_t stride)
{
  int bsize = 2;
  dim3 block(bsize, bsize);

  int size_x = std::ceil((float)width / bsize);
  int size_y = std::ceil((float)height / bsize);
  printf("width %d %d \n", width, height);
  printf("a %d b %d\n", size_x, size_y);
  dim3 grid(size_x, size_y);

  if (!type.compare("dilation"))
    kernel_dilation<<<grid, block>>>(dst_device, src_device, structuring_radius, width, height);
  else if (!type.compare("erosion"))
    kernel_erosion<<<grid, block>>>(dst_device, src_device, structuring_radius, width, height);
  else
  {
    unsigned char *tmp_device;
    cudaMallocHost((void**)&tmp_device, width * height * sizeof(unsigned char *));

    if (!type.compare("dilation_sep"))
    {
      kernel_dilation_sep_1<<<grid, block>>>(tmp_device, src_device, structuring_radius, width, height);
      cudaError_t cudaerr = cudaDeviceSynchronize();
      kernel_dilation_sep_2<<<grid, block>>>(dst_device, tmp_device, structuring_radius, width, height);

    }
    else if (!type.compare("erosion_sep"))
    {
      kernel_erosion_sep_1<<<grid, block>>>(tmp_device, src_device, structuring_radius, width, height);
      cudaError_t cudaerr = cudaDeviceSynchronize();
      kernel_erosion_sep_2<<<grid, block>>>(dst_device, tmp_device, structuring_radius, width, height);
    }
    else if (!type.compare("erosion_sep_shared"))
    {
      int tile_w = 1;
      int tile_h = height;
      dim3 block_shared_y(tile_w, tile_h);
      dim3 grid_shared_y(width, 1);
      kernel_erosion_sep_1_shared<<<grid_shared_y, block_shared_y, tile_w*tile_h*sizeof(unsigned char)>>>(tmp_device, src_device, structuring_radius, width, height, tile_w, tile_h);
      cudaError_t cudaerr = cudaDeviceSynchronize();
      
      tile_w = width;
      tile_h = 1;
      dim3 block_shared_x(tile_w, tile_h);
      dim3 grid_shared_x(1, height);
      kernel_erosion_sep_2_shared<<<grid_shared_x, block_shared_x, tile_w*tile_h*sizeof(unsigned char)>>>(dst_device, tmp_device, structuring_radius, width, height, tile_w, tile_h);

    }
    else if (!type.compare("dilation_sep_shared"))
    {
      int tile_w = 1;
      int tile_h = height;
      dim3 block_shared_y(tile_w, tile_h);
      dim3 grid_shared_y(width, 1);
      kernel_dilation_sep_1_shared<<<grid_shared_y, block_shared_y, tile_w*tile_h*sizeof(unsigned char)>>>(tmp_device, src_device, structuring_radius, width, height, tile_w, tile_h);
      cudaError_t cudaerr = cudaDeviceSynchronize();

      tile_w = width;
      tile_h = 1;
      dim3 block_shared_x(tile_w, tile_h);
      dim3 grid_shared_x(1, height);
      kernel_dilation_sep_2_shared<<<grid_shared_x, block_shared_x, tile_w*tile_h*sizeof(unsigned char)>>>(dst_device, tmp_device, structuring_radius, width, height, tile_w, tile_h);
    }

  }

  cudaError_t cudaerr = cudaDeviceSynchronize();
  //waits for all kernels to run. If there is an error in the kernel it will show here

}
