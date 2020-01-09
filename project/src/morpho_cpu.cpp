#define STB_IMAGE_IMPLEMENTATION

#include <cstddef>
#include <memory>
#include <png.h>
#include <iostream>
#include <fstream>
#include "render.hpp"
#include "../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"

void write_png(const std::byte* buffer,
    int width,
    int height,
    int stride,
    const char* filename)
{
  png_structp png_ptr =
    png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr)
    return;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, nullptr);
    return;
  }

  FILE* fp = fopen(filename, "wb");
  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr,
      width,
      height,
      8,
      PNG_COLOR_TYPE_RGB_ALPHA,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);
  for (int i = 0; i < height; ++i)
  {
    png_write_row(png_ptr, reinterpret_cast<png_const_bytep>(buffer));
    buffer += stride;
  }

  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, nullptr);
  fclose(fp);
}


void dilatation_cpu(char* hostBuffer, unsigned char* image, int width, int height)
{
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
          if (image[i*pitch + j] > 0)
          {
            //printf("found\n");
            buffer[y*pitch + x] = 255;  
            stop = true;
          }
        }
      }
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

  int w, h, bpp;

  // STBI_grey -> 1 channel
  unsigned char* rgb_image = stbi_load(argv[1], &w, &h, &bpp, STBI_grey);
  // Create buffer
  constexpr int kRGBASize = 4;
  int stride = w * kRGBASize;
  auto buffer = std::make_unique<std::byte[]>(h * stride);


  int blacks = 0;
  int whites = 0;
  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      if (rgb_image[i*w+j] == 0)
      {
        ++blacks;
      }
      else
      {
        rgb_image[i*w+j] = 255;
        ++whites;
      }
    }
  }

  printf("found %d black and %d white pixels.\n", blacks, whites);

  dilatation_cpu(reinterpret_cast<char*>(buffer.get()), rgb_image, w, h);

  stbi_image_free(rgb_image); 

  int channel_number = 1; // write new image with 1 channel
  stbi_write_png("output.png", w, h, channel_number, buffer.get(), w);
  spdlog::info("Output saved in {}.", "output.png");
  
  return 0;
}

