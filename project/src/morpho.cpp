#define STB_IMAGE_IMPLEMENTATION

#include <cstddef>
#include <memory>
#include <png.h>
#include <iostream>
#include <fstream>
#include "render.hpp"
#include <ctime>
#include "image_processor.hh"



// Usage: ./mandel
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

  int stride = 1;
  if (!type.compare("dilation"))
    dilation(buffer, uc_image, width, height, stride);
  //else
  //  erosion(buffer, uc_image, width, height);

  clock_t end = std::clock();
  std::cout << "Time: " << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

  array_to_file(buffer, dst, height, width);
}

