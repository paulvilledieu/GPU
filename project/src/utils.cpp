#include "utils.hh"


void FIBITMAP_to_uc(FIBITMAP* image, unsigned char* buffer, int width, int height)
{
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      RGBQUAD val;
      FreeImage_GetPixelColor(image, x, y, &val);
      buffer[y*width+x] = val.rgbRed;
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
