#pragma once

#include <iostream>
#include <fstream>
#include <math.h>

unsigned char *file_to_array(std::string filename, unsigned int size);
void array_to_file(unsigned char *image, std::string filename, unsigned int height, unsigned int width);
float L2_dist(int ax, int bx, int ay, int by);
