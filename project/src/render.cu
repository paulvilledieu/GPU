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
  int end_x = x+structuring_radius > width ? width : x+structuring_radius;
  int end_y = y+structuring_radius > height ? height : y+structuring_radius;
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
  int end_x = x+structuring_radius > width ? width : x+structuring_radius;
  int end_y = y+structuring_radius > height ? height : y+structuring_radius;
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


void run_kernel(std::string type, unsigned char* dst_device, unsigned char* src_device,
                int width, int height, int structuring_radius, std::ptrdiff_t stride)
{
  // Allocate device memory
  int bsize = 2;
  dim3 block(bsize, bsize);

  // returns the best pitch (padding) which is the new row

  int size_x = std::ceil((float)width / bsize);
  int size_y = std::ceil((float)height / bsize);
  printf("width %d %d \n", width, height);
  printf("a %d b %d\n", size_x, size_y);
  dim3 grid(size_x, size_y);

  if (!type.compare("dilation"))
    kernel_dilation<<<grid, block>>>(dst_device, src_device, structuring_radius, width, height);
  else
    kernel_erosion<<<grid, block>>>(dst_device, src_device, structuring_radius, width, height);


  cudaError_t cudaerr = cudaDeviceSynchronize();
  //waits for all kernels to run. If there is an error in the kernel it will show here

}
