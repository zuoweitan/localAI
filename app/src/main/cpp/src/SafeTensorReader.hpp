#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

struct TensorInfo {
  std::string dtype;
  std::vector<int> shape;
  std::vector<long> data_offsets;
};

class SafeTensorReader {
 private:
  std::string filename_;
  std::map<std::string, TensorInfo> tensor_map_;
  long header_size_;

  float fp16_to_fp32(uint16_t fp16_val) {
    uint32_t sign = (fp16_val & 0x8000) << 16;
    uint32_t exponent = (fp16_val & 0x7C00) >> 10;
    uint32_t mantissa = fp16_val & 0x03FF;

    if (exponent == 0) {
      if (mantissa == 0) {
        return *reinterpret_cast<float*>(&sign);
      } else {
        exponent = 127 - 15 + 1;
        while ((mantissa & 0x400) == 0) {
          mantissa <<= 1;
          exponent--;
        }
        mantissa &= 0x3FF;
        uint32_t fp32_bits = sign | (exponent << 23) | (mantissa << 13);
        return *reinterpret_cast<float*>(&fp32_bits);
      }
    } else if (exponent == 0x1F) {
      uint32_t fp32_bits = sign | 0x7F800000 | (mantissa << 13);
      return *reinterpret_cast<float*>(&fp32_bits);
    } else {
      exponent = exponent - 15 + 127;
      uint32_t fp32_bits = sign | (exponent << 23) | (mantissa << 13);
      return *reinterpret_cast<float*>(&fp32_bits);
    }
  }

  uint16_t fp32_to_fp16(float fp32_val) {
    uint32_t fp32_bits = *reinterpret_cast<uint32_t*>(&fp32_val);
    uint32_t sign = (fp32_bits & 0x80000000) >> 16;
    uint32_t exponent = (fp32_bits & 0x7F800000) >> 23;
    uint32_t mantissa = fp32_bits & 0x007FFFFF;

    if (exponent == 0) {
      return static_cast<uint16_t>(sign);
    } else if (exponent == 0xFF) {
      uint16_t fp16_bits =
          static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
      return fp16_bits;
    } else {
      int32_t new_exponent = static_cast<int32_t>(exponent) - 127 + 15;
      if (new_exponent <= 0) {
        return static_cast<uint16_t>(sign);
      } else if (new_exponent >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00);
      } else {
        uint16_t fp16_bits = static_cast<uint16_t>(sign | (new_exponent << 10) |
                                                   (mantissa >> 13));
        return fp16_bits;
      }
    }
  }

  void parse_header() {
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + filename_);
    }

    uint64_t header_size_raw;
    file.read(reinterpret_cast<char*>(&header_size_raw), 8);
    if (file.gcount() != 8) {
      throw std::runtime_error("Cannot read header size");
    }
    header_size_ = static_cast<long>(header_size_raw);

    std::vector<char> header_buffer(header_size_);
    file.read(header_buffer.data(), header_size_);
    if (file.gcount() != static_cast<std::streamsize>(header_size_)) {
      throw std::runtime_error("Cannot read header");
    }

    std::string header_str(header_buffer.begin(), header_buffer.end());
    nlohmann::json header_json;
    try {
      header_json = nlohmann::json::parse(header_str);
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }

    for (auto& [tensor_name, tensor_info] : header_json.items()) {
      if (tensor_name == "__metadata__") {
        continue;
      }

      TensorInfo info;
      info.dtype = tensor_info["dtype"];
      info.shape = tensor_info["shape"].get<std::vector<int>>();
      info.data_offsets = tensor_info["data_offsets"].get<std::vector<long>>();

      tensor_map_[tensor_name] = info;
    }

    file.close();
  }

  int calculate_tensor_size(const std::vector<int>& shape) {
    int size = 1;
    for (int dim : shape) {
      size *= dim;
    }
    return size;
  }

 public:
  std::vector<float> data;
  std::vector<uint16_t> fp16_data;

  explicit SafeTensorReader(const std::string& filename)
      : filename_(filename), header_size_(0) {
    parse_header();
  }

  bool read(const std::string& tensor_name, bool convert = true) {
    auto it = tensor_map_.find(tensor_name);
    if (it == tensor_map_.end()) {
      throw std::runtime_error("Tensor not found: " + tensor_name);
    }

    const TensorInfo& info = it->second;

    if (info.dtype != "F16" && info.dtype != "F32" && info.dtype != "F64") {
      throw std::runtime_error("Unsupported tensor dtype: " + info.dtype);
    }

    int tensor_size = calculate_tensor_size(info.shape);
    long data_start = info.data_offsets[0];
    long data_end = info.data_offsets[1];

    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + filename_);
    }

    file.seekg(8 + header_size_ + data_start);

    if (info.dtype == "F16") {
      long expected_bytes = static_cast<long>(tensor_size) * 2;
      if (data_end - data_start != expected_bytes) {
        throw std::runtime_error("Data size mismatch for tensor: " +
                                 tensor_name);
      }

      fp16_data.resize(tensor_size);
      file.read(reinterpret_cast<char*>(fp16_data.data()), expected_bytes);
      if (file.gcount() != static_cast<std::streamsize>(expected_bytes)) {
        throw std::runtime_error("Cannot read tensor data: " + tensor_name);
      }

      if (convert) {
        data.resize(tensor_size);
        for (int i = 0; i < tensor_size; ++i) {
          data[i] = fp16_to_fp32(fp16_data[i]);
        }
      }
    } else if (info.dtype == "F32") {
      long expected_bytes = static_cast<long>(tensor_size) * 4;
      if (data_end - data_start != expected_bytes) {
        throw std::runtime_error("Data size mismatch for tensor: " +
                                 tensor_name);
      }

      data.resize(tensor_size);
      file.read(reinterpret_cast<char*>(data.data()), expected_bytes);
      if (file.gcount() != static_cast<std::streamsize>(expected_bytes)) {
        throw std::runtime_error("Cannot read tensor data: " + tensor_name);
      }

      fp16_data.resize(tensor_size);
      for (int i = 0; i < tensor_size; ++i) {
        fp16_data[i] = fp32_to_fp16(data[i]);
      }
    } else if (info.dtype == "F64") {
      long expected_bytes = static_cast<long>(tensor_size) * 8;
      if (data_end - data_start != expected_bytes) {
        throw std::runtime_error("Data size mismatch for tensor: " +
                                 tensor_name);
      }

      std::vector<double> fp64_data(tensor_size);
      file.read(reinterpret_cast<char*>(fp64_data.data()), expected_bytes);
      if (file.gcount() != static_cast<std::streamsize>(expected_bytes)) {
        throw std::runtime_error("Cannot read tensor data: " + tensor_name);
      }

      data.resize(tensor_size);
      fp16_data.resize(tensor_size);
      for (int i = 0; i < tensor_size; ++i) {
        data[i] = static_cast<float>(fp64_data[i]);
        fp16_data[i] = fp32_to_fp16(data[i]);
      }
    }

    file.close();
    return true;
  }

  bool has_tensor(const std::string& tensor_name) const {
    return tensor_map_.find(tensor_name) != tensor_map_.end();
  }

  std::vector<int> get_tensor_shape(const std::string& tensor_name) const {
    auto it = tensor_map_.find(tensor_name);
    if (it != tensor_map_.end()) {
      return it->second.shape;
    }
    return {};
  }

  std::vector<std::string> get_tensor_names() const {
    std::vector<std::string> names;
    for (const auto& pair : tensor_map_) {
      names.push_back(pair.first);
    }
    return names;
  }

  int get_tensor_count() const { return tensor_map_.size(); }
};