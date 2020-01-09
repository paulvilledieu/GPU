#define STB_IMAGE_IMPLEMENTATION

#include <cstddef>
#include <memory>
#include <png.h>
#include <iostream>
#include <fstream>
#include "FreeImage.h"
#include "render.hpp"

#define BPP 24

void dilatation_cpu(unsigned char* buffer, unsigned char* image, int width, int height)
{
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      buffer[y*height+x] = image[y*height+x];
    }
  }

  /*
     int structuring_radius = 3;
     bool stop = false;

     for (int y = 0; y < height && !stop; ++y)
     {
     for (int x = 0; x < width && !stop; ++x)
     {
     int start_x = x-structuring_radius < 0 ? 0 : x-structuring_radius;
     int start_y = y-structuring_radius < 0 ? 0 : y-structuring_radius;
     int end_x = x+structuring_radius >= width ? width : x+structuring_radius;
     int end_y = y+structuring_radius >= height ? height : y+structuring_radius;

     for (int i = start_y; i < end_y && !stop; ++i)
     {
  //printf("%d  %d\n", threadIdx.x, threadIdx.y);
  for (int j = start_x; j < end_x && !stop; ++j)
  {
  //printf("%d\n", (int)image[i*pitch + j]);
  if (image[i*height + j] > 0)
  {
  //printf("found\n");
  buffer[y*height + x] = 255;  
  stop = true;
  }
  }
  }
  }
  }
   */
}

void FIBITMAP_to_uc(FIBITMAP* image, unsigned char* buffer, int width, int height)
{
  
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      RGBQUAD val;
      buffer[y*height+x] = FreeImage_GetPixelColor(image, x, y, &val);
    }
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

  FIBITMAP* image = FreeImage_Load(FIF_PNG, argv[1], PNG_DEFAULT);
  
  int width = FreeImage_GetWidth(image);
  int height = FreeImage_GetHeight(image);
  
  unsigned char* uc_image = (unsigned char*)malloc(width*height*sizeof(unsigned char));
  FIBITMAP_to_uc(image, uc_image, width, height);

  unsigned char* buffer;
  dilatation_cpu(buffer, uc_image, width, height);

  FIBITMAP* result = FreeImage_Allocate(width, height, BPP);
  
  if (!result)
  {
    std::cout << "Could not allocate new image." << std::endl;
    exit(1);
  }

  uc_to_FIBITMAP(buffer, result, width, height);
  
  if (FreeImage_Save(FIF_PNG, result, "output.png", 0))
    std::cout << "Image output.png saved." << std::endl;

  return 0;
}

