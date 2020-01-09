#include "FreeImage.h"
#include <iostream>

#define BPP 24

FIBITMAP* write_img(unsigned char *buffer, unsigned width, unsigned height, std::string filename)
{
  FreeImage_Initialise();
  FIBITMAP* bitmap = FreeImage_Allocate(width, height, BPP);
  RGBQUAD color;

 if (!bitmap)
    exit(1);

 for (unsigned i = 0; i < width; ++i)
 {
   for (unsigned j = 0; j < height; ++j)
   {
     unsigned char v = buffer[j*height+i];
     color.rgbRed = v;
     color.rgbGreen = v;
     color.rgbBlue = v;
     FreeImage_SetPixelColor(bitmap, i, j, &color);
   }
 }

 if (FreeImage_Save(FIF_PNG, bitmap, filename.c_str(), 0))
   std::cout << "Image " << filename << " saved." << std::endl;


  FreeImage_DeInitialise();
}
