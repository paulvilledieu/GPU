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
__global__ void mykernel(char* buffer, unsigned char *image, int structuring_radius,
                        int width, int height, size_t pitch)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
  int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
  int end_x = x+structuring_radius >= width ? width : x+structuring_radius;
  int end_y = y+structuring_radius >= height ? height : y+structuring_radius;
  bool stop = false;

  printf("HERE\n");
  //printf("%d   %d\n", start_x, end_x);
  //printf("%d   %d\n", start_y, end_y);
  for (int i = start_x; i < end_x && !stop; ++i)
  {
    //printf("%d  %d\n", threadIdx.x, threadIdx.y);
    for (int j = start_y; j < end_y && !stop; ++j)
    {
      printf("%d\n", j*pitch+i);
      //printf("%c\n", image[j*pitch + i]);
      if (image[j*pitch + i] > 0)
      {
        printf("found\n");
        buffer[y*pitch + x] = 255;  
        stop = true;
      }
    }
  }
}

void dilatation(char* hostBuffer, unsigned char* image, int width, int height, std::ptrdiff_t stride)
{
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  size_t pitch;
  char* devBuffer;
  int bsize = 32;
  dim3 block(bsize, bsize);

  // returns the best pitch (padding) which is the new row
  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
  if (rc)
    abortError("Fail buffer allocation");


  int structuring_radius = 11;
  int a = std::ceil((float)width / bsize);
  int b = std::ceil((float)height / bsize);
  printf("zidth %d %d \n", width, height);
  printf("a %d b %d\n", a, b);
  dim3 grid(a, b);

  
  spdlog::debug("running kernel of size ({},{})", width, height);
  
  mykernel<<<grid, block>>>(devBuffer, image, structuring_radius, width, height, pitch);
  
  if (cudaPeekAtLastError())
    abortError("Computation Error");
  
  // Copy back to host
  cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
  
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  //rc = cudaFree(devBuffer);
  //if (rc)
  //  abortError("Unable to free memory");
}

//n must be odd
