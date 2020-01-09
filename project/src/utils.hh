#pragma once
#define _GLIBCXX_USE_CXX11_ABI 0 
#include "FreeImage.h"
#include <iostream>

void FIBITMAP_to_uc(FIBITMAP* image, unsigned char* buffer, int width, int height);

void uc_to_FIBITMAP(unsigned char* buffer, FIBITMAP* image, int width, int height);
