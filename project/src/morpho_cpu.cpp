#include <cstddef>
#include <memory>
#include <png.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <iomanip>
#include "render.hpp"
#include "image_processor.hh"


void dilation_cpu(unsigned char* buffer, unsigned char* image, int width, int height, bool is_square, int structuring_radius)
{
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {

      bool stop = false;
      int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
      int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
      int end_x = x+structuring_radius > width - 1 ? width - 1 : x+structuring_radius;
      int end_y = y+structuring_radius > height - 1 ? height - 1: y+structuring_radius;

      for (int i = start_y; i <= end_y && !stop; ++i)
      {
        for (int j = start_x; j <= end_x && !stop; ++j)
        {
          if ((is_square && (int)image[i*width + j] == 0) || (!is_square && L2_dist(x, j, y, i) <= structuring_radius && (int)image[i*width+j] == 0))
          {
            buffer[y*width + x] = 0;
            stop = true;
          }
        }
      }
      if (!stop)
        buffer[y*width+x] = 255;
    }
  }
}

void erosion_cpu(unsigned char* buffer, unsigned char* image, int width, int height, bool is_square, int structuring_radius)
{


  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {

      bool stop = false;
      int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
      int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
      int end_x = x+structuring_radius > width - 1 ? width - 1 : x+structuring_radius;
      int end_y = y+structuring_radius > height - 1 ? height - 1 : y+structuring_radius;

      for (int i = start_y; i <= end_y && !stop; ++i)
      {
        for (int j = start_x; j <= end_x && !stop; ++j)
        {
          if ((int)image[i*width + j] == 255)
          {
            buffer[y*width + x] = 255;
            stop = true;
          }
        }
      }
      if (!stop)
        buffer[y*width+x] = 0;
    }
  }
}



int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  if (argc < 8)
  {
    std::cout << "Missing args: <operation> <src> <width> <height> <structuring_type> <structuring_radius> <dst>" << std::endl;
    return 1;
  }

  std::string type = argv[1];
  std::string src = argv[2];
  int width = std::stoi(argv[4]);
  int height = std::stoi(argv[3]);
  std::string dst = argv[7];
  std::string structuring_type = argv[5];
  int structuring_radius = std::stoi(argv[6]);

  std::cout << width << " " << height << std::endl;
  
  unsigned char *uc_image = file_to_array(src, width * height);
  
  unsigned char* buffer = (unsigned char *)malloc(width*height*sizeof(unsigned char));

  std::clock_t c_start = std::clock();

  if (!type.compare("dilation"))
    dilation_cpu(buffer, uc_image, width, height, !structuring_type.compare("square"), structuring_radius);
  else
    erosion_cpu(buffer, uc_image, width, height, !structuring_type.compare("square"), structuring_radius);

  std::clock_t c_end = std::clock();
  float cpu_time = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  std::cout << std::fixed << std::setprecision(2) << "CPU time used: "
              << cpu_time << " ms\n";
   
  if (!type.compare("dilation"))
  {
    std::ofstream bench_file;
    std::string bench_file_name = "bench/bench_cpu_" + type + ".csv";
    bench_file.open(bench_file_name, std::ios_base::app); // append instead of overwrite
    bench_file << src << ", " << cpu_time << std::endl;
  }
 

  array_to_file(buffer, dst, height, width);
}

