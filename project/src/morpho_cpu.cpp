#include <cstddef>
#include <memory>
#include <png.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include "render.hpp"
#include "image_processor.hh"


void dilation_cpu(unsigned char* buffer, unsigned char* image, int width, int height)
{

  int structuring_radius = 1;

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {

      bool stop = false;
      int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
      int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
      int end_x = x+structuring_radius > width ? width : x+structuring_radius;
      int end_y = y+structuring_radius > height ? height : y+structuring_radius;

      for (int i = start_y; i <= end_y && !stop; ++i)
      {
        for (int j = start_x; j <= end_x && !stop; ++j)
        {
          if ((int)image[i*width + j] == 0)
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

void erosion_cpu(unsigned char* buffer, unsigned char* image, int width, int height)
{

  int structuring_radius = 1;

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {

      bool stop = false;
      int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
      int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
      int end_x = x+structuring_radius > width ? width : x+structuring_radius;
      int end_y = y+structuring_radius > height ? height : y+structuring_radius;

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

  if (argc < 6)
  {
    std::cout << "Missing args: <operation> <src> <width> <height> <dst>" << std::endl;
    return 1;
  }

  std::string type = argv[1];
  std::string src = argv[2];
  int width = std::stoi(argv[4]);
  int height = std::stoi(argv[3]);
  std::string dst = argv[5];

  std::cout << width << " " << height << std::endl;
  
  unsigned char *uc_image = file_to_array(src, width * height);
  
  unsigned char* buffer = (unsigned char *)malloc(width*height*sizeof(unsigned char));

  std::clock_t start;

  if (!type.compare("dilation"))
    dilation_cpu(buffer, uc_image, width, height);
  else
    erosion_cpu(buffer, uc_image, width, height);

  clock_t end = std::clock();
  std::cout << "Time: " << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

  array_to_file(buffer, dst, height, width);
}

