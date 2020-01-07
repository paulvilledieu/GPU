#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

// Device code
__global__ void mykernel(char* buffer, char *image, int structuring_radius,
                        int width, int height)
{
  printf("here\n");
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  printf("fouadfasdfasdfand\n");
  int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
  int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
  int end_x = x+structuring_radius >= width ? width : x+structuring_radius;
  int end_y = y+structuring_radius >= height ? height : y+structuring_radius;
  bool stop = false;
  for (int i = start_x; i < end_x && !stop; ++i)
  {
    for (int j = start_y; j < end_y && !stop; ++j)
    {
        printf("%f\n", image[i*width + j]);
      if (image[i*width + j] > 0)
      {
        printf("found\n");
        buffer[x*width + y] = 255;  
        stop = true;
      }
    }
  }
}

void dilatation(char* hostBuffer, char* image, int width, int height)
{
  //cudaError_t rc = cudaSuccess;

  // Allocate device memory
  //char*  devBuffer;
  //why not use pitch?
  //size_t pitch;
  char* devBuffer;
  int bsize = 32;
  dim3 block(bsize, bsize);
   
  int structuring_radius = 11;
  auto a = std::ceil((float)width / bsize);
  auto b = std::ceil((float)height / bsize);
  printf("zidth %d %d \n", width, height);
  printf("a %d b %d\n", a, b);
  dim3 grid(a, b);
  mykernel<<<grid, block>>>(devBuffer, image, structuring_radius, width, height);
  
  // Copy back to main memory
  cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
  
  /*
  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
  if (rc)
    abortError("Fail buffer allocation");

  // Run the kernel with blocks of size 64 x 64
  {
    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);
    mykernel<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch);

    if (cudaPeekAtLastError())
      abortError("Computation Error");
  }

  // Copy back to main memory
  rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(devBuffer);
  if (rc)
    abortError("Unable to free memory");
  */
}

//n must be odd
