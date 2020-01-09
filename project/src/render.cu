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

struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

// Executed on device, called by host
__global__ void mykernel(unsigned char* buffer, unsigned char *image, int structuring_radius,
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

   
 
  //printf("%d\n", (int)image[0]);
  /*for (int i = start_y; i < end_y && !stop; ++i)
  {
    printf("%d  %d\n", threadIdx.x, threadIdx.y);
    for (int j = start_x; j < end_x && !stop; ++j)
    {
      printf("%d\n", (int)image[i*pitch + j]);
      if (image[i*pitch + j] > 0)
      {
        printf("found\n");
        buffer[y*pitch + x] = 255;  
        stop = true;
      }
    }
  }*/

  //if (!stop)
  if (x % 1000 == 0)
    printf("%d\n", image[y*pitch+x]);
  //printf("%d   %d   %d\n", x, y, y*pitch+x);
  buffer[y*pitch+x] = image[y*pitch+x];
}

void dilatation(char* hostBuffer, unsigned char* image, int width, int height, std::ptrdiff_t stride)
{
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  size_t pitch;
  unsigned char* devBuffer;
  int bsize = 32;
  dim3 block(bsize, bsize);

  // returns the best pitch (padding) which is the new row
  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(unsigned char), height);
  if (rc)
    abortError("Fail buffer allocation");


  int structuring_radius = 3;
  int a = std::ceil((float)width / bsize);
  int b = std::ceil((float)height / bsize);
  printf("width %d %d \n", width, height);
  printf("a %d b %d\n", a, b);
  dim3 grid(a, b);

  //spdlog::debug("running kernel of size ({},{})", width, height);

  // copy host data to device before calling the kernel
  unsigned char* image_device;
  cudaMalloc((void**)&image_device, width*height*sizeof(unsigned char));
  cudaMemcpy2D(image_device, width*sizeof(unsigned char), image, pitch, width*sizeof(unsigned char), height, cudaMemcpyHostToDevice);
  //cudaMemcpy(image_device, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

  mykernel<<<grid, block>>>(devBuffer, image_device, structuring_radius, width, height, pitch);
  cudaError_t cudaerr = cudaDeviceSynchronize();
  //waits for all kernels to run. If there is an error in the kernel it will show here
  
  if (cudaPeekAtLastError())
    abortError("Computation Error");
  
  // Copy back to host
  cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);
  
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(devBuffer);
  if (rc)
    abortError("Unable to free memory");
}

//n must be odd
