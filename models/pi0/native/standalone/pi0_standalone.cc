#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "opencv2/opencv.hpp"
#include "proto/msg.pb.h"
#include "utils/model_manager.h"

namespace {

using xlm::utils::FlushType;
using xlm::utils::InferBackend;
using xlm::utils::InferParam;
using xlm::utils::ModelInfo;
using xlm::utils::ModelManager;

constexpr int kCameraSlots = 3;
constexpr int kRealCameraCount = 2;
constexpr int kSiglipTokens = 256;
constexpr int kEmbeddingSize = 2048;
constexpr int kKvCount = 36;
constexpr int kKvHeadSize = 256;
constexpr int kActionHorizon = 50;
constexpr int kModelActionSize = 32;
constexpr int kRobotActionSize = 6;
constexpr int kGripperActionIndex = kRobotActionSize - 1;
constexpr double kGripperActionMin = 0.0;
constexpr double kGripperActionMax = 100.0;
constexpr int kExpertSuffixSize = 51;
constexpr int kDenoiseSteps = 10;
constexpr int kPromptTokenCount = 16;
constexpr int kPhysicalVisionTokens = kCameraSlots * kSiglipTokens;
constexpr int kValidVisionTokens = kRealCameraCount * kSiglipTokens;
constexpr int kPromptCapacity = 48;
constexpr int kPrefixSize = kPhysicalVisionTokens + kPromptCapacity;
constexpr int kExpertInputWidth = kPrefixSize + kExpertSuffixSize;
constexpr int kImageSize = 224;
constexpr uint32_t kMaxMessageBytes = 64U * 1024U * 1024U;
constexpr float kMaskValue = -32767.0f;

struct ImageInput {
  int height = 0;
  int width = 0;
  std::vector<uint8_t> chw;
};

struct RequestInput {
  uint32_t sequence = 0;
  bool reset = false;
  std::array<ImageInput, kRealCameraCount> images;
  std::array<double, kRobotActionSize> state{};
  std::string task;
};

struct EngineConfig {
  std::filesystem::path siglip_hbm;
  std::filesystem::path paligemma_hbm;
  std::filesystem::path expert_hbm;
  std::filesystem::path prompt_embedding;
  std::filesystem::path norm_stats;
  std::filesystem::path fixed_noise;
  std::filesystem::path dump_directory;
  std::string server_ip;
  int server_port = 30001;
  std::string task;
  bool parallel_siglip = true;
  std::array<int, kCameraSlots> siglip_cores{0, 1, 2};
  std::vector<int> paligemma_cores{0};
  std::vector<int> expert_cores{0, 1, 2, 3};
};

struct NormStats {
  std::array<double, kRobotActionSize> state_mean{};
  std::array<double, kRobotActionSize> state_std{};
  std::array<double, kRobotActionSize> action_mean{};
  std::array<double, kRobotActionSize> action_std{};
};

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("Could not open " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

std::vector<uint8_t> ReadBinary(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Could not open " + path.string());
  }
  stream.seekg(0, std::ios::end);
  const auto size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  if (size < 0) {
    throw std::runtime_error("Could not size " + path.string());
  }
  std::vector<uint8_t> data(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char*>(data.data()), data.size());
  if (!stream) {
    throw std::runtime_error("Could not read " + path.string());
  }
  return data;
}

void WriteBinary(const std::filesystem::path& path, const void* data,
                 size_t size) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Could not write " + path.string());
  }
  stream.write(static_cast<const char*>(data), size);
}

void MaybeWriteBinary(const std::filesystem::path& directory,
                      const std::string& name, const void* data, size_t size) {
  if (directory.empty()) return;
  WriteBinary(directory / name, data, size);
}

uint16_t HalfBits(float value) {
  cv::float16_t half(value);
  uint16_t bits = 0;
  static_assert(sizeof(bits) == sizeof(half));
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

size_t TensorElements(const hbDNNTensor& tensor) {
  size_t elements = 1;
  for (int index = 0; index < tensor.properties.validShape.numDimensions;
       ++index) {
    elements *= static_cast<size_t>(
        tensor.properties.validShape.dimensionSize[index]);
  }
  return elements;
}

std::vector<int32_t> TensorShape(const hbDNNTensor& tensor) {
  std::vector<int32_t> shape;
  for (int index = 0; index < tensor.properties.validShape.numDimensions;
       ++index) {
    shape.push_back(tensor.properties.validShape.dimensionSize[index]);
  }
  return shape;
}

std::string ShapeString(const hbDNNTensor& tensor) {
  std::ostringstream stream;
  stream << '[';
  const auto shape = TensorShape(tensor);
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) stream << ',';
    stream << shape[index];
  }
  stream << ']';
  return stream.str();
}

void RequireShape(const hbDNNTensor& tensor,
                  std::initializer_list<int32_t> expected,
                  const std::string& label) {
  if (TensorShape(tensor) != std::vector<int32_t>(expected)) {
    throw std::runtime_error(label + " has shape " + ShapeString(tensor));
  }
}

void RequireType(const hbDNNTensor& tensor, int32_t expected,
                 const std::string& label) {
  if (tensor.properties.tensorType != expected) {
    throw std::runtime_error(label + " has tensor type " +
                             std::to_string(tensor.properties.tensorType));
  }
}

void RequireMemory(const hbDNNTensor& tensor, size_t required,
                   const std::string& label) {
  if (tensor.sysMem.virAddr == nullptr || tensor.sysMem.memSize < required) {
    throw std::runtime_error(label + " has insufficient memory");
  }
}

InferBackend CoreBackend(int core) {
  switch (core) {
    case 0: return InferBackend::kInferBackendBPU0;
    case 1: return InferBackend::kInferBackendBPU1;
    case 2: return InferBackend::kInferBackendBPU2;
    case 3: return InferBackend::kInferBackendBPU3;
    default: throw std::runtime_error("Invalid BPU core " + std::to_string(core));
  }
}

class ModelRunner {
 public:
  void Init(const std::string& pack_name, const std::filesystem::path& hbm,
            const std::string& model_fragment, const std::vector<int>& cores) {
    manager_ = std::make_shared<ModelManager>();
    ModelInfo info;
    info.hbm_name = pack_name;
    info.hbm_path = hbm.string();
    if (manager_->Init(info) != 0) {
      throw std::runtime_error("Failed to initialize " + hbm.string());
    }
    for (const auto& candidate : manager_->model_names()) {
      if (candidate.find(model_fragment) != std::string::npos) {
        model_name_ = candidate;
        break;
      }
    }
    if (model_name_.empty()) {
      throw std::runtime_error("Model " + model_fragment + " not found");
    }
    if (manager_->GetInputTensors(model_name_, inputs_) != 0 ||
        manager_->GetOutputTensors(model_name_, outputs_) != 0 ||
        inputs_ == nullptr || outputs_ == nullptr) {
      throw std::runtime_error("Failed to obtain tensors for " + model_name_);
    }
    for (int core : cores) infer_param_.backend.push_back(CoreBackend(core));
  }

  int Run(const std::vector<int32_t>& input_ids = {},
          const std::vector<int32_t>& output_ids = {}) {
    if (manager_->FlushInputMem(model_name_, FlushType::kFlushClean,
                                input_ids) != 0) {
      return -1;
    }
    hbUCPTaskHandle_t task = nullptr;
    int result = manager_->InferTaskSync(model_name_, task, infer_param_);
    if (result == 0) {
      result = manager_->FlushOutputMem(
          model_name_, FlushType::kFlushInvalidate, output_ids);
    }
    if (task != nullptr) {
      const int release_result = manager_->ReleaseTaskHandle(task);
      if (result == 0) result = release_result;
    }
    return result;
  }

  std::vector<hbDNNTensor>& inputs() { return *inputs_; }
  std::vector<hbDNNTensor>& outputs() { return *outputs_; }

 private:
  std::shared_ptr<ModelManager> manager_;
  std::string model_name_;
  std::vector<hbDNNTensor>* inputs_ = nullptr;
  std::vector<hbDNNTensor>* outputs_ = nullptr;
  InferParam infer_param_;
};

std::vector<int> ParseCores(const nlohmann::json& config,
                            const std::string& key,
                            const std::vector<int>& defaults) {
  if (!config.contains(key)) return defaults;
  return config.at(key).get<std::vector<int>>();
}

EngineConfig LoadConfig(const std::filesystem::path& path) {
  const auto config = nlohmann::json::parse(ReadText(path));
  EngineConfig result;
  result.siglip_hbm = config.at("siglip_hbm_path").get<std::string>();
  result.paligemma_hbm = config.at("paligemma_hbm_path").get<std::string>();
  result.expert_hbm = config.at("action_hbm_path").get<std::string>();
  result.norm_stats = config.at("norm_stats_path").get<std::string>();
  result.server_ip = config.at("server_ip").get<std::string>();
  result.server_port = config.at("server_port").get<int>();
  result.task = config.value(
      "task", "Place the RDK camera box on top of the black MCU box.");
  result.parallel_siglip = config.value("parallel_siglip", true);
  if (config.value("real_camera_num", 0) != kRealCameraCount ||
      config.value("exec_size", 0) != kRobotActionSize ||
      config.value("denoise_num", 0) != kDenoiseSteps) {
    throw std::runtime_error("Expected real_camera_num=2 exec_size=6 denoise_num=10");
  }
  const auto siglip_cores = ParseCores(config, "siglip_bpu_core", {0, 1, 2});
  if (siglip_cores.size() != kCameraSlots) {
    throw std::runtime_error("siglip_bpu_core must contain three entries");
  }
  std::copy(siglip_cores.begin(), siglip_cores.end(), result.siglip_cores.begin());
  result.paligemma_cores = ParseCores(config, "paligemma_bpu_core", {0});
  result.expert_cores = ParseCores(config, "action_bpu_core", {0, 1, 2, 3});
  result.prompt_embedding = config.contains("prompt_embedding_path")
      ? std::filesystem::path(config.at("prompt_embedding_path").get<std::string>())
      : result.paligemma_hbm.parent_path() / "fixed_prompt_embedding.bin";
  if (const char* fixed_noise = std::getenv("PI0_FIXED_NOISE_FILE");
      fixed_noise != nullptr && fixed_noise[0] != '\0') {
    result.fixed_noise = fixed_noise;
  } else if (config.contains("fixed_noise_path")) {
    result.fixed_noise = config.at("fixed_noise_path").get<std::string>();
  }
  if (const char* dump_directory = std::getenv("PI0_STANDALONE_DUMP_DIR");
      dump_directory != nullptr && dump_directory[0] != '\0') {
    result.dump_directory = dump_directory;
  } else if (config.contains("dump_directory")) {
    result.dump_directory = config.at("dump_directory").get<std::string>();
  }
  return result;
}

template <size_t Size>
std::array<double, Size> JsonArray(const nlohmann::json& value,
                                   const std::string& label) {
  const auto source = value.get<std::vector<double>>();
  if (source.size() != Size) {
    throw std::runtime_error(label + " must contain " + std::to_string(Size));
  }
  std::array<double, Size> result{};
  std::copy(source.begin(), source.end(), result.begin());
  return result;
}

NormStats LoadNormStats(const std::filesystem::path& path) {
  const auto root = nlohmann::json::parse(ReadText(path)).at("norm_stats");
  NormStats stats;
  stats.state_mean = JsonArray<kRobotActionSize>(root.at("state").at("mean"), "state mean");
  stats.state_std = JsonArray<kRobotActionSize>(root.at("state").at("std"), "state std");
  stats.action_mean = JsonArray<kRobotActionSize>(root.at("actions").at("mean"), "action mean");
  stats.action_std = JsonArray<kRobotActionSize>(root.at("actions").at("std"), "action std");
  return stats;
}

class Pi0Engine {
 public:
  explicit Pi0Engine(EngineConfig config);
  std::array<double, kActionHorizon * kRobotActionSize> Infer(
      const RequestInput& request);

 private:
  void ValidateLayout();
  void PrepareStaticInputs();
  bool ValidPrefixToken(int index) const;
  std::filesystem::path RequestDumpDirectory();
  void PrepareImages(const RequestInput& request,
                     const std::filesystem::path& dump);
  void PreprocessImage(const ImageInput& image, hbDNNTensor& tensor);
  void RunSiglip();
  void CopySiglipToPaligemma(const std::filesystem::path& dump);
  void CopyKvToExpert(const std::filesystem::path& dump);
  void PrepareState(const std::array<double, kRobotActionSize>& raw_state,
                    const std::filesystem::path& dump);
  void PrepareNoise(const std::filesystem::path& dump);
  void RunExpert(const std::filesystem::path& dump);
  std::array<double, kActionHorizon * kRobotActionSize> PostprocessActions(
      const std::filesystem::path& dump);

  EngineConfig config_;
  NormStats stats_;
  std::array<std::unique_ptr<ModelRunner>, kCameraSlots> siglip_;
  ModelRunner paligemma_;
  ModelRunner expert_;
  int prompt_capacity_ = 0;
  int vision_capacity_ = 0;
  int prefix_size_ = 0;
  std::vector<uint16_t> fixed_noise_;
  std::mt19937 rng_;
  std::filesystem::path dump_directory_;
  size_t request_index_ = 0;
};

Pi0Engine::Pi0Engine(EngineConfig config)
    : config_(std::move(config)),
      stats_(LoadNormStats(config_.norm_stats)),
      rng_(std::random_device{}()),
      dump_directory_(config_.dump_directory) {
  for (int index = 0; index < kCameraSlots; ++index) {
    siglip_[index] = std::make_unique<ModelRunner>();
    siglip_[index]->Init("pi0_siglip", config_.siglip_hbm, "siglip",
                         {config_.siglip_cores[index]});
  }
  paligemma_.Init("pi0_paligemma", config_.paligemma_hbm, "gemma",
                  config_.paligemma_cores);
  expert_.Init("pi0_action", config_.expert_hbm, "gemma_expert",
               config_.expert_cores);
  ValidateLayout();
  PrepareStaticInputs();
  if (!dump_directory_.empty()) {
    std::filesystem::create_directories(dump_directory_);
  }
  std::cout << "Standalone Pi0 Engine ready: two real cameras, one masked slot, "
            << kPromptTokenCount << " prompt tokens, " << kDenoiseSteps
            << " Expert steps" << std::endl;
}

void Pi0Engine::ValidateLayout() {
  for (int index = 0; index < kCameraSlots; ++index) {
    auto& inputs = siglip_[index]->inputs();
    auto& outputs = siglip_[index]->outputs();
    if (inputs.size() != 2 || outputs.size() != 1) {
      throw std::runtime_error("SigLIP tensor count mismatch");
    }
    RequireShape(inputs[0], {1, 3, kImageSize, kImageSize}, "SigLIP image");
    RequireType(inputs[0], HB_DNN_TENSOR_TYPE_F16, "SigLIP image");
    RequireShape(inputs[1], {1, kSiglipTokens}, "SigLIP position IDs");
    RequireType(inputs[1], HB_DNN_TENSOR_TYPE_S64, "SigLIP position IDs");
    RequireShape(outputs[0], {1, kSiglipTokens, kEmbeddingSize},
                 "SigLIP output");
    RequireType(outputs[0], HB_DNN_TENSOR_TYPE_F16, "SigLIP output");
  }

  auto& pali_inputs = paligemma_.inputs();
  auto& pali_outputs = paligemma_.outputs();
  if (pali_inputs.size() != 3 || pali_outputs.size() < kKvCount + 1) {
    throw std::runtime_error("PaliGemma tensor count mismatch");
  }
  RequireShape(pali_inputs[0], {1, kPromptCapacity, kEmbeddingSize},
               "PaliGemma prompt embedding");
  RequireShape(pali_inputs[1], {1, kPhysicalVisionTokens, kEmbeddingSize},
               "PaliGemma vision embedding");
  RequireShape(pali_inputs[2], {1, 1, kPrefixSize, kPrefixSize},
               "PaliGemma attention mask");
  for (int index = 0; index < 3; ++index) {
    RequireType(pali_inputs[index], HB_DNN_TENSOR_TYPE_F16,
                "PaliGemma input " + std::to_string(index));
  }
  for (int index = 0; index < kKvCount; ++index) {
    RequireShape(pali_outputs[index + 1], {1, kPrefixSize, kKvHeadSize},
                 "PaliGemma KV " + std::to_string(index));
    RequireType(pali_outputs[index + 1], HB_DNN_TENSOR_TYPE_F16,
                "PaliGemma KV " + std::to_string(index));
  }

  auto& expert_inputs = expert_.inputs();
  auto& expert_outputs = expert_.outputs();
  if (expert_inputs.size() != kKvCount + 5 || expert_outputs.size() != 1) {
    throw std::runtime_error("Expert tensor count mismatch");
  }
  RequireShape(expert_inputs[0], {1, kModelActionSize}, "Expert state");
  RequireShape(expert_inputs[1], {1, kActionHorizon, kModelActionSize},
               "Expert noise/action");
  RequireShape(expert_inputs[2], {1}, "Expert denoise index");
  RequireShape(expert_inputs[3], {1, 1, kExpertSuffixSize, kExpertInputWidth},
               "Expert attention mask");
  RequireShape(expert_inputs[4], {1, kExpertSuffixSize},
               "Expert position IDs");
  RequireType(expert_inputs[0], HB_DNN_TENSOR_TYPE_F16, "Expert state");
  RequireType(expert_inputs[1], HB_DNN_TENSOR_TYPE_F16, "Expert noise/action");
  RequireType(expert_inputs[2], HB_DNN_TENSOR_TYPE_S32,
              "Expert denoise index");
  RequireType(expert_inputs[3], HB_DNN_TENSOR_TYPE_F16,
              "Expert attention mask");
  RequireType(expert_inputs[4], HB_DNN_TENSOR_TYPE_S32,
              "Expert position IDs");
  for (int index = 0; index < kKvCount; ++index) {
    RequireShape(expert_inputs[index + 5], {1, kPrefixSize, kKvHeadSize},
                 "Expert KV " + std::to_string(index));
    RequireType(expert_inputs[index + 5], HB_DNN_TENSOR_TYPE_F16,
                "Expert KV " + std::to_string(index));
  }
  RequireShape(expert_outputs[0], {1, kActionHorizon, kModelActionSize},
               "Expert output");
  RequireType(expert_outputs[0], HB_DNN_TENSOR_TYPE_F16, "Expert output");

  prompt_capacity_ = pali_inputs[0].properties.validShape.dimensionSize[1];
  vision_capacity_ = pali_inputs[1].properties.validShape.dimensionSize[1];
  prefix_size_ = pali_inputs[2].properties.validShape.dimensionSize[2];
}

void Pi0Engine::PrepareStaticInputs() {
  auto& pali_inputs = paligemma_.inputs();
  const auto prompt = ReadBinary(config_.prompt_embedding);
  const size_t prompt_bytes = TensorElements(pali_inputs[0]) * sizeof(uint16_t);
  if (prompt.size() != prompt_bytes) {
    throw std::runtime_error("Prompt embedding size mismatch: " +
                             std::to_string(prompt.size()) + " vs " +
                             std::to_string(prompt_bytes));
  }
  RequireMemory(pali_inputs[0], prompt_bytes, "PaliGemma prompt embedding");
  std::memcpy(pali_inputs[0].sysMem.virAddr, prompt.data(), prompt.size());

  for (int camera = 0; camera < kCameraSlots; ++camera) {
    auto& position_tensor = siglip_[camera]->inputs()[1];
    const size_t position_bytes = kSiglipTokens * sizeof(int64_t);
    RequireMemory(position_tensor, position_bytes, "SigLIP position IDs");
    auto* positions = static_cast<int64_t*>(position_tensor.sysMem.virAddr);
    std::iota(positions, positions + kSiglipTokens, int64_t{0});
  }

  auto& pali_mask_tensor = pali_inputs[2];
  const size_t pali_mask_elements = static_cast<size_t>(kPrefixSize) * kPrefixSize;
  RequireMemory(pali_mask_tensor, pali_mask_elements * sizeof(uint16_t),
                "PaliGemma attention mask");
  auto* pali_mask = static_cast<uint16_t*>(pali_mask_tensor.sysMem.virAddr);
  const uint16_t masked = HalfBits(kMaskValue);
  const uint16_t visible = HalfBits(0.0f);
  std::fill(pali_mask, pali_mask + pali_mask_elements, masked);
  for (int row = 0; row < kPrefixSize; ++row) {
    if (!ValidPrefixToken(row)) continue;
    for (int column = 0; column < kPrefixSize; ++column) {
      if (ValidPrefixToken(column)) {
        pali_mask[static_cast<size_t>(row) * kPrefixSize + column] = visible;
      }
    }
  }

  auto& expert_inputs = expert_.inputs();
  auto& expert_mask_tensor = expert_inputs[3];
  const size_t expert_mask_elements =
      static_cast<size_t>(kExpertSuffixSize) * kExpertInputWidth;
  RequireMemory(expert_mask_tensor, expert_mask_elements * sizeof(uint16_t),
                "Expert attention mask");
  auto* expert_mask = static_cast<uint16_t*>(expert_mask_tensor.sysMem.virAddr);
  std::fill(expert_mask, expert_mask + expert_mask_elements, visible);
  for (int row = 0; row < kExpertSuffixSize; ++row) {
    auto* row_data = expert_mask + static_cast<size_t>(row) * kExpertInputWidth;
    for (int column = 0; column < kPrefixSize; ++column) {
      if (!ValidPrefixToken(column)) row_data[column] = masked;
    }
  }
  std::fill(expert_mask + kPrefixSize + 1,
            expert_mask + kExpertInputWidth, masked);

  auto& position_tensor = expert_inputs[4];
  RequireMemory(position_tensor, kExpertSuffixSize * sizeof(int32_t),
                "Expert position IDs");
  auto* positions = static_cast<int32_t*>(position_tensor.sysMem.virAddr);
  std::iota(positions, positions + kExpertSuffixSize,
            kValidVisionTokens + kPromptTokenCount);

  if (!config_.fixed_noise.empty()) {
    const auto noise = ReadBinary(config_.fixed_noise);
    const size_t expected = kActionHorizon * kModelActionSize * sizeof(uint16_t);
    if (noise.size() != expected) {
      throw std::runtime_error("Fixed noise size mismatch: " +
                               std::to_string(noise.size()));
    }
    fixed_noise_.resize(expected / sizeof(uint16_t));
    std::memcpy(fixed_noise_.data(), noise.data(), noise.size());
    std::cout << "Using fixed Expert noise from " << config_.fixed_noise
              << std::endl;
  }
}

bool Pi0Engine::ValidPrefixToken(int index) const {
  if (index >= 0 && index < kValidVisionTokens) return true;
  const int prompt_begin = kPhysicalVisionTokens;
  return index >= prompt_begin && index < prompt_begin + kPromptTokenCount;
}

std::filesystem::path Pi0Engine::RequestDumpDirectory() {
  if (dump_directory_.empty()) return {};
  std::ostringstream name;
  name << "request_" << std::setw(6) << std::setfill('0') << request_index_++;
  const auto path = dump_directory_ / name.str();
  std::filesystem::create_directories(path);
  return path;
}

void Pi0Engine::PreprocessImage(const ImageInput& image, hbDNNTensor& tensor) {
  if (image.height <= 0 || image.width <= 0 ||
      image.chw.size() != static_cast<size_t>(3) * image.height * image.width) {
    throw std::runtime_error("Invalid CHW image");
  }
  RequireMemory(tensor, 3 * kImageSize * kImageSize * sizeof(uint16_t),
                "SigLIP image");

  std::vector<cv::Mat> channels;
  channels.reserve(3);
  for (int channel = 0; channel < 3; ++channel) {
    channels.emplace_back(
        image.height, image.width, CV_8UC1,
        const_cast<uint8_t*>(image.chw.data()) +
            static_cast<size_t>(channel) * image.height * image.width);
  }
  cv::Mat source;
  cv::merge(channels, source);
  const float ratio = std::max(static_cast<float>(image.width) / kImageSize,
                               static_cast<float>(image.height) / kImageSize);
  const int resized_width = std::max(1, static_cast<int>(std::round(image.width / ratio)));
  const int resized_height =
      std::max(1, static_cast<int>(std::round(image.height / ratio)));
  cv::Mat resized;
  cv::resize(source, resized, cv::Size(resized_width, resized_height), 0, 0,
             cv::INTER_LINEAR);
  const int pad_width = kImageSize - resized_width;
  const int pad_height = kImageSize - resized_height;
  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, pad_height / 2,
                     pad_height - pad_height / 2, pad_width / 2,
                     pad_width - pad_width / 2, cv::BORDER_CONSTANT,
                     cv::Scalar(0, 0, 0));
  cv::split(padded, channels);

  auto* output = static_cast<uint16_t*>(tensor.sysMem.virAddr);
  const size_t plane_size = static_cast<size_t>(kImageSize) * kImageSize;
  for (int channel = 0; channel < 3; ++channel) {
    const auto* source_plane = channels[channel].ptr<uint8_t>();
    for (size_t index = 0; index < plane_size; ++index) {
      output[static_cast<size_t>(channel) * plane_size + index] =
          HalfBits(static_cast<float>(source_plane[index]) * (2.0f / 255.0f) -
                   1.0f);
    }
  }
}

void Pi0Engine::PrepareImages(const RequestInput& request,
                              const std::filesystem::path& dump) {
  for (int camera = 0; camera < kRealCameraCount; ++camera) {
    auto& tensor = siglip_[camera]->inputs()[0];
    PreprocessImage(request.images[camera], tensor);
    MaybeWriteBinary(dump, "siglip_" + std::to_string(camera) + "_image_fp16.bin",
                     tensor.sysMem.virAddr,
                     3 * kImageSize * kImageSize * sizeof(uint16_t));
  }
  auto& empty = siglip_[kCameraSlots - 1]->inputs()[0];
  const size_t elements = 3 * kImageSize * kImageSize;
  RequireMemory(empty, elements * sizeof(uint16_t), "Empty SigLIP image");
  std::fill_n(static_cast<uint16_t*>(empty.sysMem.virAddr), elements,
              HalfBits(-1.0f));
  MaybeWriteBinary(dump, "siglip_2_image_fp16.bin", empty.sysMem.virAddr,
                   elements * sizeof(uint16_t));
}

void Pi0Engine::RunSiglip() {
  const auto started = std::chrono::steady_clock::now();
  auto run = [this](int camera) {
    if (siglip_[camera]->Run({}, {0}) != 0) {
      throw std::runtime_error("SigLIP inference failed for camera " +
                               std::to_string(camera));
    }
  };
  if (config_.parallel_siglip) {
    std::array<std::future<void>, kCameraSlots> futures;
    for (int camera = 0; camera < kCameraSlots; ++camera) {
      futures[camera] = std::async(std::launch::async, run, camera);
    }
    for (auto& future : futures) future.get();
  } else {
    for (int camera = 0; camera < kCameraSlots; ++camera) run(camera);
  }
  const double elapsed = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  std::cout << "SigLIP: " << elapsed << " ms" << std::endl;
}

void Pi0Engine::CopySiglipToPaligemma(
    const std::filesystem::path& dump) {
  auto& vision = paligemma_.inputs()[1];
  const size_t camera_bytes =
      static_cast<size_t>(kSiglipTokens) * kEmbeddingSize * sizeof(uint16_t);
  RequireMemory(vision, kCameraSlots * camera_bytes,
                "PaliGemma vision embedding");
  auto* destination = static_cast<uint8_t*>(vision.sysMem.virAddr);
  for (int camera = 0; camera < kCameraSlots; ++camera) {
    auto& output = siglip_[camera]->outputs()[0];
    RequireMemory(output, camera_bytes, "SigLIP output");
    std::memcpy(destination + static_cast<size_t>(camera) * camera_bytes,
                output.sysMem.virAddr, camera_bytes);
    MaybeWriteBinary(dump,
                     "siglip_" + std::to_string(camera) + "_output_fp16.bin",
                     output.sysMem.virAddr, camera_bytes);
  }
  MaybeWriteBinary(dump, "paligemma_prompt_fp16.bin",
                   paligemma_.inputs()[0].sysMem.virAddr,
                   TensorElements(paligemma_.inputs()[0]) * sizeof(uint16_t));
  MaybeWriteBinary(dump, "paligemma_vision_fp16.bin", vision.sysMem.virAddr,
                   kCameraSlots * camera_bytes);
  MaybeWriteBinary(dump, "paligemma_attention_mask_fp16.bin",
                   paligemma_.inputs()[2].sysMem.virAddr,
                   TensorElements(paligemma_.inputs()[2]) * sizeof(uint16_t));
}

void Pi0Engine::CopyKvToExpert(const std::filesystem::path& dump) {
  auto& pali_outputs = paligemma_.outputs();
  auto& expert_inputs = expert_.inputs();
  const size_t kv_bytes =
      static_cast<size_t>(kPrefixSize) * kKvHeadSize * sizeof(uint16_t);
  for (int index = 0; index < kKvCount; ++index) {
    auto& source = pali_outputs[index + 1];
    auto& destination = expert_inputs[index + 5];
    RequireMemory(source, kv_bytes, "PaliGemma KV");
    RequireMemory(destination, kv_bytes, "Expert KV");
    std::memcpy(destination.sysMem.virAddr, source.sysMem.virAddr, kv_bytes);
    if (!dump.empty()) {
      std::ostringstream name;
      name << "expert_kv_" << std::setw(2) << std::setfill('0') << index
           << "_fp16.bin";
      MaybeWriteBinary(dump, name.str(), destination.sysMem.virAddr, kv_bytes);
    }
  }
}

void Pi0Engine::PrepareState(
    const std::array<double, kRobotActionSize>& raw_state,
    const std::filesystem::path& dump) {
  auto& tensor = expert_.inputs()[0];
  const size_t bytes = kModelActionSize * sizeof(uint16_t);
  RequireMemory(tensor, bytes, "Expert state");
  auto* state = static_cast<uint16_t*>(tensor.sysMem.virAddr);
  std::fill(state, state + kModelActionSize, HalfBits(0.0f));
  for (int index = 0; index < kRobotActionSize; ++index) {
    if (!std::isfinite(raw_state[index])) {
      throw std::runtime_error("Robot state contains NaN or Inf");
    }
    const double normalized =
        (raw_state[index] - stats_.state_mean[index]) /
        (stats_.state_std[index] + 1e-6);
    state[index] = HalfBits(static_cast<float>(normalized));
  }
  MaybeWriteBinary(dump, "expert_state_fp16.bin", tensor.sysMem.virAddr, bytes);
}

void Pi0Engine::PrepareNoise(const std::filesystem::path& dump) {
  auto& tensor = expert_.inputs()[1];
  const size_t elements =
      static_cast<size_t>(kActionHorizon) * kModelActionSize;
  const size_t bytes = elements * sizeof(uint16_t);
  RequireMemory(tensor, bytes, "Expert noise/action");
  auto* noise = static_cast<uint16_t*>(tensor.sysMem.virAddr);
  if (!fixed_noise_.empty()) {
    std::memcpy(noise, fixed_noise_.data(), bytes);
  } else {
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t index = 0; index < elements; ++index) {
      noise[index] = HalfBits(distribution(rng_));
    }
  }
  MaybeWriteBinary(dump, "expert_noise_fp16.bin", tensor.sysMem.virAddr, bytes);
  MaybeWriteBinary(dump, "expert_attention_mask_fp16.bin",
                   expert_.inputs()[3].sysMem.virAddr,
                   TensorElements(expert_.inputs()[3]) * sizeof(uint16_t));
  MaybeWriteBinary(dump, "expert_position_ids_i32.bin",
                   expert_.inputs()[4].sysMem.virAddr,
                   TensorElements(expert_.inputs()[4]) * sizeof(int32_t));
}

void Pi0Engine::RunExpert(const std::filesystem::path& dump) {
  auto& inputs = expert_.inputs();
  auto& output = expert_.outputs()[0];
  const size_t action_elements =
      static_cast<size_t>(kActionHorizon) * kModelActionSize;
  const size_t action_bytes = action_elements * sizeof(uint16_t);
  RequireMemory(output, action_bytes, "Expert output");
  RequireMemory(inputs[1], action_bytes, "Expert noise/action");
  const auto started = std::chrono::steady_clock::now();
  for (int step = 0; step < kDenoiseSteps; ++step) {
    std::memcpy(inputs[2].sysMem.virAddr, &step, sizeof(step));
    const std::vector<int32_t> dynamic_inputs =
        step == 0 ? std::vector<int32_t>{} : std::vector<int32_t>{1, 2};
    if (expert_.Run(dynamic_inputs, {0}) != 0) {
      throw std::runtime_error("Expert inference failed at step " +
                               std::to_string(step));
    }
    std::memcpy(inputs[1].sysMem.virAddr, output.sysMem.virAddr, action_bytes);
    if (!dump.empty()) {
      std::ostringstream name;
      name << "expert_step_" << std::setw(2) << std::setfill('0') << step
           << "_output_fp16.bin";
      MaybeWriteBinary(dump, name.str(), output.sysMem.virAddr, action_bytes);
    }
  }
  const double elapsed = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  std::cout << "Expert x" << kDenoiseSteps << ": " << elapsed << " ms"
            << std::endl;
}

std::array<double, kActionHorizon * kRobotActionSize>
Pi0Engine::PostprocessActions(const std::filesystem::path& dump) {
  auto& output = expert_.outputs()[0];
  const size_t action_bytes =
      static_cast<size_t>(kActionHorizon) * kModelActionSize * sizeof(uint16_t);
  RequireMemory(output, action_bytes, "Expert output");
  cv::Mat fp16(kActionHorizon, kModelActionSize, CV_16F,
               output.sysMem.virAddr);
  cv::Mat fp32;
  fp16.convertTo(fp32, CV_32F);
  std::array<double, kActionHorizon * kRobotActionSize> actions{};
  std::array<double, kActionHorizon * kRobotActionSize> unclipped_actions{};
  size_t clipped_gripper_steps = 0;
  double raw_gripper_min = std::numeric_limits<double>::infinity();
  double raw_gripper_max = -std::numeric_limits<double>::infinity();
  for (int row = 0; row < kActionHorizon; ++row) {
    const auto* normalized = fp32.ptr<float>(row);
    for (int column = 0; column < kRobotActionSize; ++column) {
      double value =
          static_cast<double>(normalized[column]) *
              (stats_.action_std[column] + 1e-6) +
          stats_.action_mean[column];
      if (!std::isfinite(value)) {
        throw std::runtime_error("Expert output contains NaN or Inf");
      }
      const size_t action_index =
          static_cast<size_t>(row) * kRobotActionSize + column;
      unclipped_actions[action_index] = value;
      if (column == kGripperActionIndex) {
        raw_gripper_min = std::min(raw_gripper_min, value);
        raw_gripper_max = std::max(raw_gripper_max, value);
        const double clipped_value =
            std::clamp(value, kGripperActionMin, kGripperActionMax);
        clipped_gripper_steps += static_cast<size_t>(clipped_value != value);
        value = clipped_value;
      }
      actions[action_index] = value;
    }
  }
  if (clipped_gripper_steps > 0) {
    std::cout << "Gripper clamp: raw=[" << raw_gripper_min << ", "
              << raw_gripper_max << "] clipped_steps="
              << clipped_gripper_steps << std::endl;
  }
  MaybeWriteBinary(dump, "expert_final_fp16.bin", output.sysMem.virAddr,
                   action_bytes);
  MaybeWriteBinary(dump, "actions_unclipped_fp64.bin", unclipped_actions.data(),
                   unclipped_actions.size() * sizeof(double));
  MaybeWriteBinary(dump, "actions_fp64.bin", actions.data(),
                   actions.size() * sizeof(double));
  return actions;
}

std::array<double, kActionHorizon * kRobotActionSize> Pi0Engine::Infer(
    const RequestInput& request) {
  if (!request.task.empty() && request.task != config_.task) {
    throw std::runtime_error(
        "This HBM uses a fixed prompt embedding; request task does not match config");
  }
  const auto started = std::chrono::steady_clock::now();
  const auto dump = RequestDumpDirectory();
  PrepareImages(request, dump);
  RunSiglip();
  CopySiglipToPaligemma(dump);

  std::vector<int32_t> kv_output_ids(kKvCount);
  std::iota(kv_output_ids.begin(), kv_output_ids.end(), 1);
  const auto pali_started = std::chrono::steady_clock::now();
  if (paligemma_.Run({}, kv_output_ids) != 0) {
    throw std::runtime_error("PaliGemma inference failed");
  }
  const double pali_elapsed = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - pali_started)
                                  .count();
  std::cout << "PaliGemma: " << pali_elapsed << " ms" << std::endl;

  CopyKvToExpert(dump);
  PrepareState(request.state, dump);
  PrepareNoise(dump);
  RunExpert(dump);
  auto actions = PostprocessActions(dump);
  const double elapsed = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  std::cout << "Pi0 request seq=" << request.sequence << " total=" << elapsed
            << " ms first_action=[";
  for (int index = 0; index < kRobotActionSize; ++index) {
    if (index != 0) std::cout << ',';
    std::cout << actions[index];
  }
  std::cout << "]" << std::endl;
  return actions;
}

bool ReceiveAll(int socket_fd, void* destination, size_t size) {
  auto* output = static_cast<uint8_t*>(destination);
  size_t received = 0;
  while (received < size) {
    const ssize_t result =
        ::recv(socket_fd, output + received, size - received, 0);
    if (result == 0) return false;
    if (result < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    received += static_cast<size_t>(result);
  }
  return true;
}

bool SendAll(int socket_fd, const void* source, size_t size) {
  const auto* input = static_cast<const uint8_t*>(source);
  size_t sent = 0;
  while (sent < size) {
    const ssize_t result =
        ::send(socket_fd, input + sent, size - sent, MSG_NOSIGNAL);
    if (result <= 0) {
      if (result < 0 && errno == EINTR) continue;
      return false;
    }
    sent += static_cast<size_t>(result);
  }
  return true;
}

bool ReceiveMessage(int socket_fd, MultiModalInput& message) {
  uint32_t network_size = 0;
  if (!ReceiveAll(socket_fd, &network_size, sizeof(network_size))) return false;
  const uint32_t size = ntohl(network_size);
  if (size == 0 || size > kMaxMessageBytes) {
    throw std::runtime_error("Invalid protobuf message size " +
                             std::to_string(size));
  }
  std::string payload(size, '\0');
  if (!ReceiveAll(socket_fd, payload.data(), payload.size())) return false;
  if (!message.ParseFromString(payload)) {
    throw std::runtime_error("Could not parse protobuf request");
  }
  return true;
}

bool SendMessage(int socket_fd, const MultiModalInput& message) {
  std::string payload;
  if (!message.SerializeToString(&payload)) {
    throw std::runtime_error("Could not serialize protobuf response");
  }
  if (payload.empty() || payload.size() > kMaxMessageBytes) {
    throw std::runtime_error("Invalid protobuf response size");
  }
  const uint32_t network_size = htonl(static_cast<uint32_t>(payload.size()));
  return SendAll(socket_fd, &network_size, sizeof(network_size)) &&
         SendAll(socket_fd, payload.data(), payload.size());
}

size_t TensorElementCount(const Tensor& tensor) {
  size_t elements = 1;
  if (tensor.shape_size() == 0) {
    throw std::runtime_error("Protocol tensor has no shape");
  }
  for (int index = 0; index < tensor.shape_size(); ++index) {
    const int32_t dimension = tensor.shape(index);
    if (dimension <= 0 ||
        elements > std::numeric_limits<size_t>::max() /
                       static_cast<size_t>(dimension)) {
      throw std::runtime_error("Invalid protocol tensor shape");
    }
    elements *= static_cast<size_t>(dimension);
  }
  return elements;
}

RequestInput ParseRequest(const MultiModalInput& message) {
  RequestInput request;
  request.sequence = message.header().seq();
  request.reset = message.header().reset();
  if (message.images_size() != kRealCameraCount) {
    throw std::runtime_error("Standalone Pi0 requires exactly two images");
  }
  for (int camera = 0; camera < kRealCameraCount; ++camera) {
    const Tensor& tensor = message.images(camera);
    if (tensor.dtype() != Tensor::UINT8 || tensor.shape_size() != 3 ||
        tensor.shape(0) != 3) {
      throw std::runtime_error("Image tensor must be CHW uint8 RGB");
    }
    const size_t elements = TensorElementCount(tensor);
    if (tensor.data().size() != elements) {
      throw std::runtime_error("Image tensor byte count mismatch");
    }
    request.images[camera].height = tensor.shape(1);
    request.images[camera].width = tensor.shape(2);
    request.images[camera].chw.assign(tensor.data().begin(), tensor.data().end());
  }

  if (message.languages_size() != 1 ||
      message.languages(0).dtype() != Tensor::STRING) {
    throw std::runtime_error("Request must contain one string task");
  }
  request.task = message.languages(0).data();

  if (message.states_size() != 1 ||
      message.states(0).dtype() != Tensor::FLOAT64) {
    throw std::runtime_error("Request must contain one float64 robot state");
  }
  const Tensor& state = message.states(0);
  if (TensorElementCount(state) != kRobotActionSize ||
      state.data().size() != kRobotActionSize * sizeof(double)) {
    throw std::runtime_error("Robot state must contain six float64 values");
  }
  std::memcpy(request.state.data(), state.data().data(), state.data().size());
  return request;
}

MultiModalInput BuildResponse(
    const RequestInput& request,
    const std::array<double, kActionHorizon * kRobotActionSize>& actions) {
  MultiModalInput response;
  response.mutable_header()->set_seq(request.sequence);
  response.mutable_header()->set_reset(request.reset);
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(now).count();
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count() -
      seconds * 1000000000LL;
  response.mutable_header()->mutable_stamp()->set_sec(seconds);
  response.mutable_header()->mutable_stamp()->set_nsec(
      static_cast<int32_t>(nanoseconds));
  response.mutable_header()->set_frame_id("s600_standalone_pi0");

  Tensor* tensor = response.add_states();
  tensor->set_dtype(Tensor::FLOAT64);
  tensor->add_shape(1);
  tensor->add_shape(kActionHorizon);
  tensor->add_shape(kRobotActionSize);
  tensor->set_data(reinterpret_cast<const char*>(actions.data()),
                   actions.size() * sizeof(double));
  return response;
}

int ConnectToServer(const std::string& ip, int port) {
  for (;;) {
    const int socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
      throw std::runtime_error("Could not create TCP socket");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (::inet_pton(AF_INET, ip.c_str(), &address.sin_addr) != 1) {
      ::close(socket_fd);
      throw std::runtime_error("Invalid server IP " + ip);
    }
    if (::connect(socket_fd, reinterpret_cast<sockaddr*>(&address),
                  sizeof(address)) == 0) {
      std::cout << "Connected to controller at " << ip << ':' << port
                << std::endl;
      return socket_fd;
    }
    const int error = errno;
    ::close(socket_fd);
    std::cerr << "Controller connection failed: " << std::strerror(error)
              << "; retrying" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

int RunMain(int argc, char** argv) {
  std::filesystem::path config_path;
  bool validate_only = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--config" && index + 1 < argc) {
      config_path = argv[++index];
    } else if (argument == "--validate-only") {
      validate_only = true;
    } else {
      std::cerr << "Usage: " << argv[0]
                << " --config CONFIG_JSON [--validate-only]" << std::endl;
      return 2;
    }
  }
  if (config_path.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " --config CONFIG_JSON [--validate-only]" << std::endl;
    return 2;
  }

  const EngineConfig config = LoadConfig(config_path);
  Pi0Engine engine(config);
  if (validate_only) return 0;

  for (;;) {
    const int socket_fd = ConnectToServer(config.server_ip, config.server_port);
    for (;;) {
      MultiModalInput message;
      if (!ReceiveMessage(socket_fd, message)) break;
      const RequestInput request = ParseRequest(message);
      const auto actions = engine.Infer(request);
      if (!SendMessage(socket_fd, BuildResponse(request, actions))) break;
    }
    ::close(socket_fd);
    std::cerr << "Controller disconnected; reconnecting" << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  try {
    return RunMain(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "pi0_standalone fatal: " << error.what() << std::endl;
    return 1;
  }
}
