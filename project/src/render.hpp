#pragma once
#include <cstddef>
#include <memory>





/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
void run_kernel(std::string type, unsigned char* dst_device, unsigned char *src_device, int width, int height, int structuring_radius, std::ptrdiff_t stride);

