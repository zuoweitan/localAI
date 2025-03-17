#ifndef SDUTILS_HPP
#define SDUTILS_HPP

#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize2.h"

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>

struct ClipInput
{
  std::vector<int> input_ids;
};
struct UnetInput
{
  std::vector<float> latents;
  int timestep;
  std::vector<uint16_t> text_embedding;
  std::vector<float> text_embedding_float;
};
struct UnetOutput
{
  std::vector<float> latents;
};
struct VaeDecoderInput
{
  std::vector<float> latents;
};

struct Picture
{
  std::vector<float> pixel_values;
  std::vector<uint8_t> pixels;
};

struct GenerationResult
{
  std::vector<uint8_t> image_data;
  int width;
  int height;
  int channels;
  int generation_time_ms;
  int first_step_time_ms;
};

inline std::string base64_encode(const std::string &in)
{
  static const auto lookup =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(in.size());
  auto val = 0;
  auto valb = -6;
  for (auto c : in)
  {
    val = (val << 8) + static_cast<uint8_t>(c);
    valb += 8;
    while (valb >= 0)
    {
      out.push_back(lookup[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6)
  {
    out.push_back(lookup[((val << 8) >> (valb + 8)) & 0x3F]);
  }
  while (out.size() % 4)
  {
    out.push_back('=');
  }
  return out;
}

inline unsigned hashSeed(unsigned long long seed)
{
  seed = ((seed >> 16) ^ seed) * 0x45d9f3b;
  seed = ((seed >> 16) ^ seed) * 0x45d9f3b;
  seed = (seed >> 16) ^ seed;
  return seed;
}

inline std::string LoadBytesFromFile(const std::string &path)
{
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail())
  {
    std::cerr << "Cannot open " << path << std::endl;
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}
template <typename T>
void saveVectorToFile(const std::vector<T> &vec, const std::string &filename)
{
  static_assert(std::is_trivially_copyable<T>::value,
                "Type T must be trivially copyable to be saved directly to file");
  std::ofstream file(filename, std::ios::binary);
  if (!file)
  {
    throw std::runtime_error("Cannot open file for writing");
  }
  if (!vec.empty())
  {
    file.write(reinterpret_cast<const char *>(vec.data()), vec.size() * sizeof(T));
  }
  if (!file)
  {
    throw std::runtime_error("Error writing to file");
  }
  file.close();
}
template <typename T>
std::vector<T> loadVectorFromFile(const std::string &filename)
{
  static_assert(std::is_trivially_copyable<T>::value,
                "Type T must be trivially copyable to be loaded directly from file");
  std::ifstream file(filename, std::ios::binary);
  if (!file)
  {
    throw std::runtime_error("Cannot open file for reading");
  }
  file.seekg(0, std::ios::end);
  size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  size_t elementCount = fileSize / sizeof(T);
  std::vector<T> vec(elementCount);
  if (elementCount > 0)
  {
    file.read(reinterpret_cast<char *>(vec.data()), fileSize);
    if (!file)
    {
      throw std::runtime_error("Error reading vector data");
    }
  }
  file.close();
  return vec;
}

bool safety_check(const std::vector<uint8_t> &image_data,
                  int width,
                  int height,
                  float &nsfw_score,
                  MNN::Interpreter *interpreter,
                  MNN::Session *session)
{
  try
  {
    std::vector<uint8_t> resized_256(256 * 256 * 3);
    if (!stbir_resize_uint8_linear(
            image_data.data(), width, height, 0,
            resized_256.data(), 256, 256, 0,
            STBIR_RGB))
    {
      throw std::runtime_error("Resize failed");
    }
    std::vector<unsigned char> jpeg_buffer;
    if (!stbi_write_jpg_to_func([](void *context, void *data, int size)
                                {
                auto& buffer = *static_cast<std::vector<unsigned char>*>(context);
                buffer.insert(buffer.end(), 
                        static_cast<unsigned char*>(data),
                        static_cast<unsigned char*>(data) + size); }, &jpeg_buffer, 256, 256, 3, resized_256.data(), 95))
    {
      throw std::runtime_error("JPEG encoding failed");
    }
    int jpeg_width, jpeg_height, jpeg_channels;
    uint8_t *decoded_data = stbi_load_from_memory(jpeg_buffer.data(), jpeg_buffer.size(),
                                                  &jpeg_width, &jpeg_height, &jpeg_channels, 3);
    if (!decoded_data)
    {
      throw std::runtime_error("JPEG decoding failed");
    }
    std::vector<float> processed_data(224 * 224 * 3);
    int crop_x = (256 - 224) / 2;
    int crop_y = (256 - 224) / 2;
    float vgg_mean[] = {104.0f, 117.0f, 123.0f};
    for (int y = 0; y < 224; y++)
    {
      for (int x = 0; x < 224; x++)
      {
        for (int c = 0; c < 3; c++)
        {
          int src_idx = ((y + crop_y) * 256 + (x + crop_x)) * 3 + c;
          int dst_idx = (y * 224 + x) * 3 + c;
          processed_data[dst_idx] = static_cast<float>(decoded_data[src_idx]) - vgg_mean[c];
        }
      }
    }
    stbi_image_free(decoded_data);
    auto input_tensor = interpreter->getSessionInput(session, nullptr);
    std::vector<int> dims = {1, 224, 224, 3};
    interpreter->resizeTensor(input_tensor, dims);
    interpreter->resizeSession(session);
    auto inputHost = input_tensor->host<float>();
    memcpy(inputHost, processed_data.data(), 224 * 224 * 3 * sizeof(float));
    interpreter->runSession(session);
    auto output_tensor = interpreter->getSessionOutput(session, nullptr);
    auto outputHost = output_tensor->host<float>();
    nsfw_score = outputHost[1];
    std::cout << "NSFW Score: " << nsfw_score << std::endl;
    return true;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Safety check error: " << e.what() << std::endl;
    return false;
  }
}
inline void PrintEncodeResult(const std::vector<int> &ids)
{
  std::cout << "tokens=[";
  for (size_t i = 0; i < ids.size(); ++i)
  {
    if (i != 0)
      std::cout << ", ";
    std::cout << ids[i];
  }
  std::cout << "]" << std::endl;
}
#endif