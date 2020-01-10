#include<iostream>
#include<fstream>

unsigned char *file_to_array(std::string filename, unsigned size) {
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

int main(int argc, char *argv[]) {
  if (argc < 4)
  {
    std::cerr << "Please provied a filename, a width and a height." << std::endl;
    return 1;
  }
  unsigned int width = std::stoi(argv[2]);
  unsigned int height = std::stoi(argv[3]);
  unsigned char *image = file_to_array(argv[1], width*height);
  
  free(image);
}

