#include "FreeImage.h"
#include <iostream>

int main(int argc, char *argv[])
{
  int width = 800;
  int height = 800;
  int bpp = 24;

  FreeImage_Initialise();
  FIBITMAP* bitmap = FreeImage_Allocate(width, height, bpp);
  RGBQUAD color;

  if (!bitmap)
    exit(1);

 for (int i = 0; i < width; ++i)
 {
   for (int j = 0; j < height; ++j)
   {
     color.rgbRed = 0;
     color.rgbGreen = 255;
     color.rgbBlue = 0;
     FreeImage_SetPixelColor(bitmap, i, j, &color);
   }
 }

 if (FreeImage_Save(FIF_PNG, bitmap, "green.png", 0))
   std::cout << "Saved" << std::endl;


  bitmap = FreeImage_Load(FIF_PNG, "green.png", PNG_DEFAULT);
  std::cout << FreeImage_GetWidth(bitmap) << std::endl;

  RGBQUAD val;
  FreeImage_GetPixelColor(bitmap, 0, 0, &val);
  std::cout << (int)val.rgbRed << std::endl;
  std::cout << (int)val.rgbGreen << std::endl;
  std::cout << (int)val.rgbBlue << std::endl;
  std::cout << (int)val.rgbReserved << std::endl;
  FreeImage_DeInitialise();
  return 0;
}
