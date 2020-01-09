#define STB_IMAGE_IMPLEMENTATION

#include <cstddef>
#include <memory>
#include <png.h>
#include <iostream>
#include <fstream>
#include "FreeImage.h"
#include "render.hpp"

#define BPP 24

void dilation_cpu(unsigned char* buffer, unsigned char* image, int width, int height)
{
  /*for (int y = 0; y < height; ++y)
    {
    for (int x = 0; x < width; ++x)
    {
    buffer[y*height+x] = image[y*height+x];
    }
    }*/

  int structuring_radius = 3;

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {

      bool stop = false;
      int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
      int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
      int end_x = x+structuring_radius >= width ? width : x+structuring_radius;
      int end_y = y+structuring_radius >= height ? height : y+structuring_radius;

      for (int i = start_y; i < end_y && !stop; ++i)
      {
        //printf("%d  %d\n", threadIdx.x, threadIdx.y);
        for (int j = start_x; j < end_x && !stop; ++j)
        {
          //printf("%d\n", (int)image[i*width + j]);
          if (image[i*width + j] == 0)
          {
            //printf("found\n");
            buffer[y*width + x] = 0;
            stop = true;
          }
        }
      }
      if (!stop)
        buffer[y*height+x] = 255;
    }
  }
}

void FIBITMAP_to_uc(FIBITMAP* image, unsigned char* buffer, int width, int height)
{
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      RGBQUAD val;
      FreeImage_GetPixelColor(image, x, y, &val);
      buffer[y*width+x] = val.rgbRed;
      //if ((int)buffer[y*width+x] != 0)
      //  std::cout << "1";
      //else
      //  std::cout << "0";
      std::cout << (int)buffer[y*width+x];
    }
    std::cout << std::endl;
  }
}

void uc_to_FIBITMAP(unsigned char* buffer, FIBITMAP* image, int width, int height)
{
  RGBQUAD color;
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      RGBQUAD color;
      unsigned char val = buffer[y*width+x];

      color.rgbRed = val;
      color.rgbGreen = val;
      color.rgbBlue = val;
      FreeImage_SetPixelColor(image, x, y, &color);
    }
  }
}

// Usage: ./mandel
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

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
  dilation_cpu(buffer, uc_image, width, height);

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
  return 0;
}

