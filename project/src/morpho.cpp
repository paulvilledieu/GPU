#define STB_IMAGE_IMPLEMENTATION

#include <cstddef>
#include <memory>
#include <png.h>
#include <iostream>
#include <fstream>
#include "render.hpp"
#define _GLIBCXX_USE_CXX11_ABI 0 
#include "FreeImage.h"
#include "utils.hh"
#include <ctime>


#define BPP 24

// Usage: ./mandel
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  // Rendering

  std::string mode = "GPU";
  if (mode == "CPU")
  {
    printf("CPU mode\n");
    exit(1);
  }
  else if (mode == "GPU")
  {
    if (argc < 2)
      return 1;

    FreeImage_Initialise();
    FIBITMAP* image = FreeImage_Load(FIF_BMP, argv[1], BMP_DEFAULT);
    int width = FreeImage_GetWidth(image);
    int height = FreeImage_GetHeight(image);

    unsigned char* uc_image = (unsigned char*)malloc(width*height*sizeof(unsigned char));
    FIBITMAP_to_uc(image, uc_image, width, height);
    std::cout << width << "  " << height << std::endl;

    unsigned char* buffer = (unsigned char *)malloc(width*height*sizeof(unsigned char));

    std::clock_t start;

   // erosion_cpu(buffer, uc_image, width, height);

    clock_t end = std::clock();
    std::cout << "Time: " << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    FIBITMAP* result = FreeImage_Allocate(width, height, BPP);

    if (!result)
    {
      std::cout << "Could not allocate new image." << std::endl;
      exit(1);
    }

    uc_to_FIBITMAP(buffer, result, width, height);
    if (FreeImage_Save(FIF_BMP, result, "output.bmp", BMP_DEFAULT))
      std::cout << "Image output.bmp saved." << std::endl;
    FreeImage_DeInitialise();


    // Create buffer
    //auto buffer = std::make_unique<std::byte[]>(h * stride);
    //    dilatation(reinterpret_cast<char*>(buffer.get()), rgb_image, w, h, stride);

  }

  // Save
  //write_png(buffer.get(), width, height, stride, filename.c_str());

  std::cout << "Output saved in output_gpu" << std::endl;
}

