#pragma once

#include<iostream>
#include<fstream>

unsigned char *file_to_array(std::string filename, unsigned int size);
void array_to_file(unsigned char *image, std::string filename, unsigned int height, unsigned int width);