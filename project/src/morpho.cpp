#define STB_IMAGE_IMPLEMENTATION

#include <cstddef>
#include <memory>
#include <png.h>
#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
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


// Usage: ./mandel
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

   CLI::App app{"morpho"};
  /*
  app.add_option("-o", filename, "Output image");
  app.add_option("width", width, "width of the output image");
  app.add_option("height", height, "height of the output image");
  app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");

  CLI11_PARSE(app, argc, argv);
  */
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

    int w, h, bpp;

    unsigned char* rgb_image = stbi_load(argv[1], &w, &h, &bpp, 3);
    // Create buffer
    constexpr int kRGBASize = 4;
    int stride = w * kRGBASize;
    auto buffer = std::make_unique<std::byte[]>(h * stride);

    spdlog::info("Runing {} mode with (w={},h={}).", mode, w, h);
    
    dilatation(reinterpret_cast<char*>(buffer.get()), rgb_image, w, h, stride);
    
    stbi_image_free(rgb_image); 

    stbi_write_png("output.png", w, h, 3, buffer.get(), w*3);
  }

  // Save
  //write_png(buffer.get(), width, height, stride, filename.c_str());
 
  spdlog::info("Output saved in {}.", "output.png");
}

