#include<iostream>
#include<fstream>

unsigned char *file_to_array(std::string filename, unsigned int size) {
  std::fstream file;
  std::string word;
  file.open(filename.c_str());
  unsigned char *result = (unsigned char*)malloc(size * sizeof(unsigned char));
  unsigned int i = 0;
  while(file >> word && i < size) { //take word and print
    result[i] = std::stoi(word);
    std::cout << (int)result[i] << std::endl;
    ++i;
  }
  file.close();
  return result;
}

void array_to_file(unsigned char *image, std::string filename, unsigned int height, unsigned int width)
{
  std::ofstream ofs(filename);
  for (int i = 0; i < height; ++i)
  {
    for (int j = 0; j < width; ++j)
    {
      ofs << (int)image[i*width+j] << " ";
    }
    ofs << std::endl;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 4)
  {
    std::cerr << "Please provied a filename, a height and a width." << std::endl;
    return 1;
  }
  unsigned int width = std::stoi(argv[3]);
  unsigned int height = std::stoi(argv[2]);
  unsigned char *image = file_to_array(argv[1], width*height);

  array_to_file(image, "mini_out.txt", 3, 4);
  free(image);
}

