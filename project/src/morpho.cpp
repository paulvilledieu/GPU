#define STB_IMAGE_IMPLEMENTATION

#include <cstddef>
#include <memory>
#include <png.h>
#include <iostream>
#include <fstream>
#include "render.hpp"
#include <ctime>
#include "image_processor.hh"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iomanip>

// Usage: ./mandel
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  if (argc < 7)
  {
    std::cout << "Missing args: <operation> <src> <width> <height> <structuring radius> <dst>" << std::endl;
    return 1;
  }

  std::string type = argv[1];
  std::string src = argv[2];
  int width = std::stoi(argv[4]);
  int height = std::stoi(argv[3]);
  int structuring_radius = std::stoi(argv[5]);
  std::string dst = argv[6];

  std::cout << width << " " << height << std::endl;

  unsigned char *uc_image = file_to_array(src, width * height);

  int size = width * height * sizeof(unsigned char);

  //unsigned char* buffer = (unsigned char *)malloc(size);


  int stride = 1;

  unsigned char *src_device;
  unsigned char *dst_device;
  unsigned char *src_host;
  unsigned char *dst_host;

  cudaMalloc((void**)&src_device, size);
  cudaMalloc((void**)&dst_device, size);
  cudaMallocHost((void**)&src_host, size);
  cudaMallocHost((void**)&dst_host, size);

  cudaMemcpy(src_host, uc_image, size, cudaMemcpyHostToHost);
  cudaMemcpy(src_device, src_host, size, cudaMemcpyHostToDevice);


  std::clock_t c_start = std::clock();
  run_kernel(type, dst_device, src_device, width, height, structuring_radius, stride);
  std::clock_t c_end = std::clock();
  float cpu_time = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  std::cout << std::fixed << std::setprecision(2) << "CPU time used: "
              << cpu_time << " ms\n";
  cudaMemcpy(dst_host, dst_device, size, cudaMemcpyDeviceToHost);

  array_to_file(dst_host, dst, height, width);
}

