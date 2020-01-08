#include <cstddef>
#include <memory>

#include <png.h>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <fstream>
#include "render.hpp"


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

void read_png_file(char *filename) {
  FILE *fp = fopen(filename, "rb");

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(!png) abort();

  png_infop info = png_create_info_struct(png);
  if(!info) abort();

  if(setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);

  png_read_info(png, info);

  int width      = png_get_image_width(png, info);
  int height     = png_get_image_height(png, info);
  png_byte color_type = png_get_color_type(png, info);
  png_byte bit_depth  = png_get_bit_depth(png, info);

  // Read any color_type into 8bit depth, RGBA format.
  // See http://www.libpng.org/pub/png/libpng-manual.txt

  if(bit_depth == 16)
    png_set_strip_16(png);

  if(color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);

  // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
  if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png);

  if(png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  // These color_type don't have an alpha channel then fill it with 0xff.
  if(color_type == PNG_COLOR_TYPE_RGB ||
     color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

  if(color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png);

  png_read_update_info(png, info);

  png_bytep *row_pointers = NULL;

  if (row_pointers) abort();

  row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
  for(int y = 0; y < height; y++) {
    row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
  }

  png_read_image(png, row_pointers);

  for (int i = 0; i < 10; ++i)
    printf("%c\n", *(png + i)); 
  fclose(fp);

  png_destroy_read_struct(&png, &info, NULL);
}

// Usage: ./mandel
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  std::string filename = "output.png";
  std::string mode = "GPU";
  int width = 215;
  int height = 167;

  CLI::App app{"morpho"};
  /*
  app.add_option("-o", filename, "Output image");
  app.add_option("width", width, "width of the output image");
  app.add_option("height", height, "height of the output image");
  app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");

  CLI11_PARSE(app, argc, argv);
  */
  // Create buffer
  constexpr int kRGBASize = 4;
  int stride = width * kRGBASize;
  auto buffer = std::make_unique<std::byte[]>(height * stride);

  // Rendering
  spdlog::info("Runnging {} mode with (w={},h={}).", mode, width, height);
  if (mode == "CPU")
  {
    printf("CPU mode\n");
    exit(1);
  }
  else if (mode == "GPU")
  {
    read_png_file("../resources/fb.png");
    std::streampos size;
    char* memblock;
    if (argc < 2)
      return 1;

    std::ifstream file (argv[1], std::ios::in|std::ios::binary|std::ios::ate);
    if (file.is_open())
    {
      size = file.tellg();
      memblock = new char [size];
      file.seekg (0, std::ios::beg);
      file.read (memblock, size);
      file.close();

      //dilatation(reinterpret_cast<char*>(buffer.get()), memblock, width, height, stride);
      //delete[] memblock;
    }
  }

  // Save
  write_png(buffer.get(), width, height, stride, filename.c_str());
 
  spdlog::info("Output saved in {}.", filename);
}

