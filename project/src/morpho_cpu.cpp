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


struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};



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


void dilatation_cpu(rgba8_t* buffer, rgba8_t* image, int width, int height)
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
  //auto buffer = std::make_unique<std::byte[]>(h * stride);
  rgba8_t* buffer = (rgba8_t*)malloc(w*h*sizeof(struct rgba8_t));
  printf("h: %d, w: %d\n", h, w); 

  


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

  rgba8_t* new_image = (rgba8_t*)malloc(w*h*sizeof(struct rgba8_t));

  for (int i = 0; i < h; ++i)
  {
    for (int j = 0; j < w; ++j)
    {
      unsigned char val = rgb_image[i*h+j];
      new_image[i*h+j] = rgba8_t{val, val, val, 255};
    }
  }

//  dilatation_cpu(reinterpret_cast<rgba8_t*>(buffer.get()), new_image, w, h);
  dilatation_cpu(buffer, new_image, w, h);

  //stbi_image_free(rgb_image); 

  int channel_number = 1; // write new image with 1 channel
  //stbi_write_png("output.png", w, h, channel_number, buffer.get(), w);
  //stbi_write_png("output.png", w, h, channel_number, buffer, w);
  write_png(buffer, w, h, stride, "output.png");
  return 0;
}

