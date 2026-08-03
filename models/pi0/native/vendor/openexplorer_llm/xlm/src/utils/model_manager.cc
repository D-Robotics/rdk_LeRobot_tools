/*
 * Copyright (C) 2024 Shanghai Gua Technology Co., Ltd.
 * All rights reserved
 */
#include "utils/model_manager.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/common.h"
#include "utils/xlm_utils.h"

// #define DUMP_MODEL_INPUT

#ifdef DUMP_MODEL_INPUT
#include <fstream>
#include <iostream>
#endif

#define TAG "mod_mgr"

// 定义llm的prefill和decode模型序号
static const int32_t kLlmPrefill = 1;
static const int32_t kLlmDecode = 0;

// 定义draft的prefill和decode模型序号
static const int32_t kDraftPrefill = 2;
static const int32_t kDraftDecode = 1;

// TaskReleaseManager 线程个数
static const int32_t kReleaseTaskThreadNum = 1;

namespace xlm::utils {

ModelManager::ModelManager() {
  hcup_flush_flags_ = {
      {FlushType::kFlushInvalidate, HB_SYS_MEM_CACHE_INVALIDATE},
      {FlushType::kFlushClean, HB_SYS_MEM_CACHE_CLEAN},
  };

  infer_backend_ = {
      {InferBackend::kInferBackendCoreAny, HB_UCP_CORE_ANY},
      {InferBackend::kInferBackendBPUAny, HB_UCP_BPU_CORE_ANY},
      {InferBackend::kInferBackendBPU0, HB_UCP_BPU_CORE_0},
      {InferBackend::kInferBackendBPU1, HB_UCP_BPU_CORE_1},
      {InferBackend::kInferBackendBPU2, HB_UCP_BPU_CORE_2},
      {InferBackend::kInferBackendBPU3, HB_UCP_BPU_CORE_3},
  };
}

void ModelManager::HbUcpTaskDoneCallBack(hbUCPTaskHandle_t taskHandle,
                                         int32_t status, void *userdata) {
  auto model_manager = static_cast<ModelManager *>(userdata);
  model_manager->task_done_callback_(taskHandle, status,
                                     model_manager->infer_model_name_);
}

std::unordered_map<std::string, std::unique_ptr<hbDNNPackedHandle_t>>
    ModelManager::packed_dnn_handle_{};  // NOLINT

std::unordered_map<std::string, std::atomic<int32_t>>
    ModelManager::pack_handle_use_count_{};  // NOLINT

std::unordered_map<
    std::string,                                                      // NOLINT
    std::unordered_map<std::string, std::shared_ptr<hbDNNHandle_t>>>  // NOLINT
    ModelManager::dnn_handles_{};                                     // NOLINT

std::shared_mutex ModelManager::packed_dnn_handle_mutex_;
std::shared_mutex ModelManager::dnn_handles_mutex_;

ModelManager::~ModelManager() {
  if (model_info_.free_tensor_mem_func != nullptr) {
    model_info_.free_tensor_mem_func(this);
  } else {
    if (model_info_.use_llm_model || model_info_.use_draft_model) {
      DeInitShared();
    } else {
      DeInit();
    }
  }

  ReleasePackedDnnHandle();
}

int32_t ModelManager::Init(ModelInfo const &model_info) {
  LOGE_AND_RETURN_IF(model_info.hbm_path.empty(), "Hbm path is empty!", -1);
  LOGE_AND_RETURN_IF(model_info.hbm_name.empty(), "Hbm name is empty!", -1);

  packed_name_ = model_info.hbm_name;
  model_info_ = model_info;

  LOGE_AND_RETURN_IF(LoadHbmPack(model_info.hbm_name, model_info.hbm_path),
                     "Load hbm pack failed!", -1);
  LOGE_AND_RETURN_IF(GetModelNames(model_info.hbm_name),
                     "Init model names failed!", -1);
  LOGE_AND_RETURN_IF(LoadDnnModel(model_info.hbm_name),
                     "Load dnn model failed!", -1);

  LOGE_AND_RETURN_IF(GetInOutputCounts(), "Get inoutput counts failed!", -1);

  if (model_info.alloc_tensor_mem_func != nullptr) {
    LOGE_AND_RETURN_IF(model_info.alloc_tensor_mem_func(this),
                       "Alloc tensor mem failed!", -1);
  } else {
    LOGE_AND_RETURN_IF(DefaultAllocTensorMem(), "Alloc tensor mem failed!", -1);
  }

  // 自动预取并缓存各子模型的编译BPU核数
  int32_t ret = GetBpuCoreNum(model_info.hbm_name);
  if (ret != 0) {
    LOGE(TAG, "GetBpuCoreNum failed for hbm: {}", model_info.hbm_name);
    return ret;
  }
  for (auto const &kv : bpu_core_num_) {
    LOGD(TAG, "compiled bpu cores: model='{}' cores={}", kv.first, kv.second);
  }

  return 0;
}

int32_t ModelManager::DeInit() {
  if (MemAllocType::kMemAllocUnique == model_info_.mem_alloc_type) {
    input_tensors_.clear();
    output_tensors_.clear();
    if (model_inout_mem_.memSize > 0) {
      hbUCPFree(&(model_inout_mem_));
      model_inout_mem_ = hbUCPSysMem{0U, nullptr, 0U};
    }
  } else if (MemAllocType::kMemAllocSharedInput == model_info_.mem_alloc_type) {
    for (auto &input_tensor : input_tensors_) {
      for (int32_t i = 0; i < model_info_.shared_input_start_index; ++i) {
        hbUCPFree(&(input_tensor.second[i].sysMem));
      }
    }

    for (int32_t i = model_info_.shared_input_start_index;
         i < input_counts_[model_names_[0]]; ++i) {
      hbUCPFree(&(input_tensors_[model_names_[0]][i].sysMem));
    }
    input_tensors_.clear();

    for (auto &output_tensor : output_tensors_) {
      for (auto &[sysMem, _] : output_tensor.second) {
        hbUCPFree(&(sysMem));
      }
    }
    output_tensors_.clear();
  }

  return 0;
}

int32_t ModelManager::DeInitShared() {
  std::string model_name_decode;
  std::string model_name_prefill;
  if (model_info_.use_llm_model) {
    model_name_decode = model_names_[kLlmDecode];
    model_name_prefill = model_names_[kLlmPrefill];
  } else if (model_info_.use_draft_model) {
    model_name_decode = model_names_[kDraftDecode];
    model_name_prefill = model_names_[kDraftPrefill];
  }

  for (auto &[sysMem, _] : input_tensors_[model_name_prefill]) {
    hbUCPFree(&(sysMem));
  }

  for (int32_t i = 0; i < model_info_.non_kv_output_count; i++) {
    hbUCPFree(&(output_tensors_[model_name_prefill][i].sysMem));
  }

  for (int32_t i = 0; i < model_info_.non_kv_input_count; i++) {
    hbUCPFree(&(input_tensors_[model_name_decode][i].sysMem));
  }

  for (int32_t i = 0; i < model_info_.non_kv_output_count; i++) {
    hbUCPFree(&(output_tensors_[model_name_decode][i].sysMem));
  }

  return 0;
}

int32_t ModelManager::LoadHbmPack(std::string const &hbm_name,
                                  std::string const &hbm_path) {
  auto model_file_name{hbm_path.c_str()};

  std::unique_lock<std::shared_mutex> packed_dnn_handle_lock(
      packed_dnn_handle_mutex_);

  if (packed_dnn_handle_.count(hbm_name) <= 0) {
    packed_dnn_handle_[hbm_name] =
        std::unique_ptr<hbDNNPackedHandle_t>(new hbDNNPackedHandle_t);
    int32_t constexpr model_file_count{1};
    LOGE_AND_RETURN_IF(
        hbDNNInitializeFromFiles(packed_dnn_handle_[hbm_name].get(),
                                 &model_file_name, model_file_count),
        "hbDNNInitializeFromFiles failed!", -1);
    LOGI(TAG, "Load hbm file '{}' success.", hbm_path);

    pack_handle_use_count_[hbm_name] = 0;
  }

  pack_handle_use_count_[hbm_name].fetch_add(1);

  return 0;
}

int32_t ModelManager::GetModelNames(std::string const &hbm_name) {
  char const **model_name_list;

  {
    std::shared_lock<std::shared_mutex> packed_dnn_handle_lock(
        packed_dnn_handle_mutex_);
    LOGE_AND_RETURN_IF(
        hbDNNGetModelNameList(&model_name_list, &model_count_,
                              *(packed_dnn_handle_[hbm_name].get())),
        "hbDNNGetModelNameList failed!", -1);
  }

  LOGI(TAG, "model_count_ is: {}", model_count_);

  model_names_.clear();
  for (size_t i{0U}; i < static_cast<size_t>(model_count_); ++i) {
    model_names_.emplace_back(std::string(model_name_list[i]));
  }

  return 0;
}

int32_t ModelManager::LoadDnnModel(std::string const &hbm_name) {
  std::shared_lock<std::shared_mutex> packed_dnn_handle_lock(
      packed_dnn_handle_mutex_);
  std::unique_lock<std::shared_mutex> dnn_handles_lock(dnn_handles_mutex_);
  if (dnn_handles_.count(hbm_name) <= 0) {
    for (auto const &model_name : model_names_) {
      dnn_handles_[hbm_name][model_name] = std::make_shared<hbDNNHandle_t>();
      LOGE_AND_RETURN_IF(
          hbDNNGetModelHandle(dnn_handles_[hbm_name][model_name].get(),
                              *(packed_dnn_handle_[hbm_name].get()),
                              model_name.c_str()),
          "hbDNNGetModelHandle failed!", -1);

      LOGI(TAG, "Load dnn model success. hbm_name is: {}, model_name is: {}",
           hbm_name, model_name);
    }
  }
  return 0;
}

int32_t ModelManager::GetBpuCoreNum(std::string const &hbm_name) {
  std::unique_lock<std::shared_mutex> dnn_handles_lock(dnn_handles_mutex_);

  for (auto const &model_name : model_names_) {
    int32_t bpu_core_num = 0;
    LOGE_AND_RETURN_IF(
        hbDNNGetCompileBpuCoreNum(&bpu_core_num,
                                  *(dnn_handles_[hbm_name][model_name].get())),
        "hbDNNGetBpuCoreNum failed!", -1);

    bpu_core_num_[model_name] = bpu_core_num;
  }

  return 0;
}

int32_t ModelManager::GetInOutputCounts() {
  for (auto const &model_name : model_names_) {
    input_counts_[model_name] = 0;
    output_counts_[model_name] = 0;

    {
      std::shared_lock<std::shared_mutex> dnn_handles_lock(dnn_handles_mutex_);
      LOGE_AND_RETURN_IF(
          hbDNNGetInputCount(
              &input_counts_[model_name],
              *(dnn_handles_[model_info_.hbm_name][model_name].get())),
          "hbDNNGetInputCount failed!", -1);

      LOGE_AND_RETURN_IF(
          hbDNNGetOutputCount(
              &output_counts_[model_name],
              *(dnn_handles_[model_info_.hbm_name][model_name].get())),
          "hbDNNGetOutputCount failed!", -1);
    }
  }

  return 0;
}

int32_t ModelManager::DefaultAllocTensorMem() {
  if (model_info_.use_llm_model || model_info_.use_draft_model) {
    if (0 == InitInOutputTensorsShared(model_info_)) {
      LOGI(TAG, "Init model success");
    } else {
      LOGE(TAG, "Init model failed!");
      return -1;
    }
  } else {
    if (MemAllocType::kMemAllocUnique == model_info_.mem_alloc_type) {
      for (size_t i{0U}; i < static_cast<size_t>(model_count_); ++i) {
        if (0 == AllocUniqueTensorMem(model_info_, model_names_[i])) {
          LOGD(TAG, "Init model[{}]: '{}' success", i, model_names_[i]);
        } else {
          LOGE(TAG, "Init model[{}]: '{}' failed!", i, model_names_[i]);
          return -1;
        }
      }
    } else if (MemAllocType::kMemAllocSharedInput ==
               model_info_.mem_alloc_type) {
      LOGE_AND_RETURN_IF(AllocSharedInputTensorMem(model_info_, model_names_),
                         "Alloc shared tensor mem failed!", -1);
    } else {
      LOGE(TAG, "alloc type {} not support!",
           static_cast<int32_t>(model_info_.mem_alloc_type));
      return -1;
    }
  }

  return 0;
}

int32_t ModelManager::AllocUniqueTensorMem(ModelInfo const &model_info,
                                           std::string const &model_name) {
  std::unordered_map<hbUCPSysMem *, uint64_t> mem2offset;
  uint64_t mem_offset = 0U;
  auto PrepareTensors = [&](std::vector<hbDNNTensor> &tensors,
                            size_t const count,
                            std::string const &tensor_type) {
    tensors.resize(count);
    int32_t mult_num = 1;

    if ("Input" == tensor_type) {
      mult_num = model_info.input_mem_multi;
    } else if ("Output" == tensor_type) {
      mult_num = model_info.output_mem_multi;
    }

    std::shared_lock<std::shared_mutex> dnn_handles_lock(dnn_handles_mutex_);
    for (size_t i{0U}; i < count; ++i) {
      if ("Input" == tensor_type) {
        LOGE_AND_RETURN_IF(
            hbDNNGetInputTensorProperties(
                &tensors[i].properties,
                *(dnn_handles_[model_info.hbm_name][model_name].get()), i),
            "hbDNNGetInputTensorProperties failed!", -1);
      } else if ("Output" == tensor_type) {
        LOGE_AND_RETURN_IF(
            hbDNNGetOutputTensorProperties(
                &tensors[i].properties,
                *(dnn_handles_[model_info.hbm_name][model_name].get()), i),
            "hbDNNGetOutputTensorProperties failed!", -1);
      } else {
        return -1;
      }
      // 计算alignedByteSize
      mem2offset[&tensors[i].sysMem] = mem_offset;
      tensors[i].sysMem.memSize =
          tensors[i].properties.alignedByteSize * mult_num;
      mem_offset += tensors[i].sysMem.memSize;
    }
    return 0;
  };

  input_tensors_[model_name] =
      std::vector<hbDNNTensor>(input_counts_[model_name]);
  output_tensors_[model_name] =
      std::vector<hbDNNTensor>(output_counts_[model_name]);

  if (PrepareTensors(input_tensors_[model_name], input_counts_[model_name],
                     "Input") ||
      PrepareTensors(output_tensors_[model_name], output_counts_[model_name],
                     "Output")) {
    return -1;
  }

  LOGE_AND_RETURN_IF(
      hbUCPMallocCached(&model_inout_mem_, mem_offset, model_info.device_id),
      "hbUCPMallocCached failed!", -1);

  for (auto &[mem_ptr, offset] : mem2offset) {
    mem_ptr->virAddr = reinterpret_cast<void *>(
        reinterpret_cast<uint64_t>(model_inout_mem_.virAddr) + offset);
    mem_ptr->phyAddr = model_inout_mem_.phyAddr + offset;
  }
  return 0;
}

int32_t ModelManager::AllocSharedInputTensorMem(
    ModelInfo const &model_info, std::vector<std::string> const &model_names) {
  for (auto const &model_name : model_names) {
    input_tensors_[model_name] =
        std::vector<hbDNNTensor>(model_info.shared_input_start_index);

    // 分配非共享的input tensor内存
    for (int32_t i = 0; i < model_info.shared_input_start_index; ++i) {
      LOGE_AND_RETURN_IF(
          hbDNNGetInputTensorProperties(
              &input_tensors_[model_name][i].properties,
              *(dnn_handles_[model_info.hbm_name][model_name].get()), i),
          "input unique hbDNNGetInputTensorProperties failed!", -1);
      LOGE_AND_RETURN_IF(
          hbUCPMallocCached(
              &input_tensors_[model_name][i].sysMem,
              input_tensors_[model_name][i].properties.alignedByteSize,
              model_info.device_id),
          "input unique hbUCPMallocCached failed!", -1);
    }

    // 分配非共享的output tensor内存
    output_tensors_[model_name] =
        std::vector<hbDNNTensor>(output_counts_[model_name]);
    for (int32_t i = 0; i < output_counts_[model_name]; ++i) {
      LOGE_AND_RETURN_IF(
          hbDNNGetOutputTensorProperties(
              &output_tensors_[model_name][i].properties,
              *(dnn_handles_[model_info.hbm_name][model_name].get()), i),
          "output unique hbDNNGetOutputTensorProperties failed!", -1);
      LOGE_AND_RETURN_IF(
          hbUCPMallocCached(
              &output_tensors_[model_name][i].sysMem,
              output_tensors_[model_name][i].properties.alignedByteSize,
              model_info.device_id),
          "output unique hbUCPMallocCached failed!", -1);
    }
  }

  // 分配共享input 内存
  std::string any_model_name = model_names_[0];  // 任意的model name即可
  for (int32_t i = model_info.shared_input_start_index;
       i < input_counts_[any_model_name]; i++) {
    hbDNNTensor shared_mem_tensor;
    hbDNNGetInputTensorProperties(
        &shared_mem_tensor.properties,
        *(dnn_handles_[model_info.hbm_name][any_model_name].get()), i);
    int32_t input_memSize = shared_mem_tensor.properties.alignedByteSize;
    hbUCPMallocCached(&shared_mem_tensor.sysMem,
                      input_memSize * model_info.input_mem_multi, 0);

    shared_mem_tensor.sysMem.memSize = input_memSize;

    for (const auto &model_name : model_names_) {
      input_tensors_[model_name].emplace_back(shared_mem_tensor);
    }
  }

  return 0;
}

int32_t ModelManager::InitInOutputTensorsShared(ModelInfo const &model_info) {
  std::string model_name_decode;
  std::string model_name_prefill;
  if (model_info_.use_llm_model) {
    model_name_decode = model_names_[kLlmDecode];
    model_name_prefill = model_names_[kLlmPrefill];
  } else if (model_info_.use_draft_model) {
    model_name_decode = model_names_[kDraftDecode];
    model_name_prefill = model_names_[kDraftPrefill];
  }

  int32_t input_count = 0;
  int32_t output_count = 0;

  LOGE_AND_RETURN_IF(
      hbDNNGetInputCount(
          &input_count,
          *(dnn_handles_[model_info.hbm_name][model_name_decode].get())),
      "hbDNNGetOutputCount failed!", -1);

  LOGE_AND_RETURN_IF(
      hbDNNGetOutputCount(
          &output_count,
          *(dnn_handles_[model_info.hbm_name][model_name_decode].get())),
      "hbDNNGetOutputCount failed!", -1);

  int32_t mult_num = 1;
  mult_num = model_info.kv_mem_multi;

  std::shared_lock<std::shared_mutex> dnn_handles_lock(dnn_handles_mutex_);

  // decode input
  for (int32_t i = 0; i < model_info.non_kv_input_count; i++) {
    hbDNNTensor input_decode;
    hbDNNGetInputTensorProperties(
        &input_decode.properties,
        *(dnn_handles_[model_info.hbm_name][model_name_decode].get()), i);
    int32_t input_memSize = input_decode.properties.alignedByteSize;
    hbUCPMallocCached(&input_decode.sysMem, input_memSize, 0);
    input_tensors_[model_name_decode].push_back(input_decode);
  }

  // decode output
  for (int32_t i = 0; i < model_info.non_kv_output_count; i++) {
    hbDNNTensor output_decode;
    hbDNNGetOutputTensorProperties(
        &output_decode.properties,
        *(dnn_handles_[model_info.hbm_name][model_name_decode].get()), i);
    int32_t output_memSize = output_decode.properties.alignedByteSize;
    hbUCPMallocCached(&output_decode.sysMem, output_memSize, 0);
    output_tensors_[model_name_decode].push_back(output_decode);
  }

  // prefill input
  for (int32_t i = 0; i < model_info.non_kv_input_count; i++) {
    hbDNNTensor input_prefill;
    hbDNNGetInputTensorProperties(
        &input_prefill.properties,
        *(dnn_handles_[model_info.hbm_name][model_name_prefill].get()), i);
    int32_t input_memSize = input_prefill.properties.alignedByteSize;
    hbUCPMallocCached(&input_prefill.sysMem, input_memSize, 0);
    input_tensors_[model_name_prefill].push_back(input_prefill);
  }

  // prefill output
  for (int32_t i = 0; i < model_info.non_kv_output_count; i++) {
    hbDNNTensor output_prefill;
    hbDNNGetOutputTensorProperties(
        &output_prefill.properties,
        *(dnn_handles_[model_info.hbm_name][model_name_prefill].get()), i);
    int32_t output_memSize = output_prefill.properties.alignedByteSize;
    hbUCPMallocCached(&output_prefill.sysMem, output_memSize, 0);
    output_tensors_[model_name_prefill].push_back(output_prefill);
  }

  // prefill decode kv cache
  for (int32_t i = model_info.non_kv_input_count; i < input_count; i++) {
    hbDNNTensor kv_mem;
    hbDNNGetInputTensorProperties(
        &kv_mem.properties,
        *(dnn_handles_[model_info.hbm_name][model_name_prefill].get()), i);
    int32_t kv_memSize = kv_mem.properties.alignedByteSize;
    hbUCPMallocCached(&kv_mem.sysMem, kv_memSize * mult_num, 0);
    kv_mem.sysMem.memSize = kv_memSize;
    input_tensors_[model_name_prefill].push_back(kv_mem);
    input_tensors_[model_name_decode].push_back(kv_mem);

    hbDNNTensor prefill_output_kv_mem;
    hbDNNGetOutputTensorProperties(
        &prefill_output_kv_mem.properties,
        *(dnn_handles_[model_info.hbm_name][model_name_prefill].get()),
        i - (model_info.non_kv_input_count - model_info.non_kv_output_count));
    int32_t prefill_output_kv_memSize =
        prefill_output_kv_mem.properties.alignedByteSize;

    kv_mem.sysMem.memSize = prefill_output_kv_memSize;
    output_tensors_[model_name_prefill].push_back(kv_mem);

    hbDNNTensor decode_output_kv_mem;
    hbDNNGetOutputTensorProperties(
        &decode_output_kv_mem.properties,
        *(dnn_handles_[model_info.hbm_name][model_name_decode].get()),
        i - (model_info.non_kv_input_count - model_info.non_kv_output_count));
    int32_t decode_output_kv_memSize =
        decode_output_kv_mem.properties.alignedByteSize;
    kv_mem.sysMem.memSize = decode_output_kv_memSize;
    output_tensors_[model_name_decode].push_back(kv_mem);
  }

  return 0;
}

int32_t ModelManager::GetInputTensors(
    std::string const &model_name, std::vector<hbDNNTensor> *&input_tensors) {
  return GetTensors(model_name, input_tensors, input_tensors_);
}

int32_t ModelManager::GetOutputTensors(
    std::string const &model_name, std::vector<hbDNNTensor> *&output_tensors) {
  return GetTensors(model_name, output_tensors, output_tensors_);
}

int32_t ModelManager::GetTensors(
    std::string const &model_name, std::vector<hbDNNTensor> *&tensors,
    std::unordered_map<std::string, std::vector<hbDNNTensor>> &tensor_storage) {
  auto iter = tensor_storage.find(model_name);
  if (tensor_storage.end() == iter) {
    LOGE("mod_mgr", "tensors not found! model_name is: {}.", model_name);
    return -1;
  }

  tensors = &iter->second;
  return 0;
}

int32_t ModelManager::FlushInputMem(
    std::string const &model_name, FlushType const &flush_type,
    std::vector<int32_t> const &flush_tensor_ids) {
  return FlushMem(model_name, flush_type, flush_tensor_ids, input_tensors_);
}

int32_t ModelManager::FlushOutputMem(
    std::string const &model_name, FlushType const &flush_type,
    std::vector<int32_t> const &flush_tensor_ids) {
  return FlushMem(model_name, flush_type, flush_tensor_ids, output_tensors_);
}

int32_t ModelManager::FlushMem(
    std::string const &model_name, FlushType const &flush_type,
    std::vector<int32_t> const &flush_tensor_ids,
    std::unordered_map<std::string, std::vector<hbDNNTensor>> &tensor_storage) {
  auto iter = tensor_storage.find(model_name);
  if (tensor_storage.end() == iter) {
    LOGE(TAG, "flush tensors not found! model_name is: {}.",
         model_name.c_str());
    return -1;
  }

  if (flush_tensor_ids.empty()) {
    for (auto const &each : iter->second) {
      LOGE_AND_RETURN_IF(
          hbUCPMemFlush(&each.sysMem, hcup_flush_flags_[flush_type]),
          "hbUCPMemFlush failed!", -1);
    }
  } else {
    for (auto const &id : flush_tensor_ids) {
      LOGE_AND_RETURN_IF(hbUCPMemFlush(&iter->second[id].sysMem,
                                       hcup_flush_flags_[flush_type]),
                         "hbUCPMemFlush failed!", -1);
    }
  }
  return 0;
}

int32_t ModelManager::InferPreSubmitTask(InferParam const &infer_param,
                                         PreSubmitParam &pre_submit_param,
                                         hbUCPTaskHandle_t &task_handle) {
  auto presubmit_infer_param = infer_param;
  pre_submit_param.infer_result = std::promise<int>();
  std::future<int> task_done_ret = pre_submit_param.infer_result.get_future();
  presubmit_infer_param.enable_pre_submit = true;

  if (task_handle == nullptr) {
    LOGE_AND_RETURN_IF(
        InferTaskAsync(pre_submit_param.dnn_model_name, task_handle,
                       presubmit_infer_param, pre_submit_param.cb),
        "InferTaskAsync failed", -1);
  }

  NotifyTask(task_handle);
  if (pre_submit_param.update_func != nullptr) {
    LOGE_AND_RETURN_IF(pre_submit_param.update_func(), "Update function failed",
                       -1);
  }

  // task_handle 触发后通过回调函数释放，因此可直接覆盖
  task_handle = nullptr;
  LOGE_AND_RETURN_IF(
      InferTaskAsync(pre_submit_param.dnn_model_name, task_handle,
                     presubmit_infer_param, pre_submit_param.cb),
      "InferTaskAsync failed", -1);

  auto ret{0};
  try {
    ret = task_done_ret.get();
    if (ret != 0) {
      LOGE(TAG, "InferTaskDoneCb failed");
    }
  } catch (const std::exception &e) {
    LOGE(TAG, "Exception in task_done_ret.get(): %s", e.what());
    ret = -1;
  } catch (...) {
    LOGE(TAG, "Unknown exception in task_done_ret.get()");
    ret = -1;
  }
  return ret;
}

int32_t ModelManager::InferTaskSync(std::string const &model_name,
                                    hbUCPTaskHandle_t &task_handle,
                                    InferParam const &infer_param,
                                    int32_t const timeout,
                                    int32_t const task_id) {
  TIME_PROFILE_BEGIN(infer_model)
  int32_t ret = (task_id == -1)
                    ? InferModel(model_name, task_handle, infer_param)
                    : InferModel(model_name, task_handle, task_id);

  LOGE_AND_RETURN_IF(ret, "InferModel for model[ " + model_name + " ] failed!",
                     -1);
  TIME_PROFILE_END(infer_model, "InferTaskSync InferModel")

  infer_model_name_ = model_name;

  TIME_PROFILE_BEGIN(submit_infer_task)
  LOGE_AND_RETURN_IF(SubmitInferTask(model_name, task_handle, infer_param),
                     "SubmitInferTask for model[" + model_name + "] failed!",
                     -1);
  TIME_PROFILE_END(submit_infer_task, "InferTaskSync SubmitInferTask")

  TIME_PROFILE_BEGIN(wait_task_done)
  LOGE_AND_RETURN_IF(hbUCPWaitTaskDone(task_handle, timeout),
                     "hbUCPWaitTaskDone for model[" + model_name + "] failed!",
                     -1);
  TIME_PROFILE_END(wait_task_done, "InferTaskSync hbUCPWaitTaskDone")
  return 0;
}

int32_t ModelManager::InferTaskAsync(std::string const &model_name,
                                     hbUCPTaskHandle_t &task_handle,
                                     InferParam const &infer_param,
                                     TaskDoneCb task_done_cb) {
  TIME_PROFILE_BEGIN(infer_model)
  LOGE_AND_RETURN_IF(InferModel(model_name, task_handle, infer_param),
                     "InferModel for model[ " + model_name + " ] failed!", -1);
  TIME_PROFILE_END(infer_model, "InferTaskAsync InferModel")

  infer_model_name_ = model_name;
  task_done_callback_ = std::move(task_done_cb);

  TIME_PROFILE_BEGIN(set_task_done_cb)
  LOGE_AND_RETURN_IF(
      hbUCPSetTaskDoneCb(task_handle, HbUcpTaskDoneCallBack, this),
      "hbUCPSetTaskDoneCb for model[" + model_name + "] failed!", -1);
  TIME_PROFILE_END(set_task_done_cb, "InferTaskAsync SetTaskDoneCb")

  TIME_PROFILE_BEGIN(submit_infer_task)
  LOGE_AND_RETURN_IF(SubmitInferTask(model_name, task_handle, infer_param),
                     "SubmitInferTask for model[" + model_name + "] failed!",
                     -1);
  TIME_PROFILE_END(submit_infer_task, "InferTaskAsync SubmitInferTask")

  return 0;
}

int32_t ModelManager::InferModel(std::string const &model_name,
                                 hbUCPTaskHandle_t &task_handle,
                                 InferParam const &infer_param) {
  std::shared_lock<std::shared_mutex> dnn_handles_lock(dnn_handles_mutex_);
#ifdef DUMP_MODEL_INPUT
  for (uint32_t i = 0; i < input_tensors_[model_name].size(); i++) {
    std::string filename = model_name + std::to_string(i) + ".bin";
    std::ofstream out_file(filename, std::ios::out | std::ios::binary);
    if (!out_file) {
      std::cerr << "Error: Cannot open file " << filename << " for writing.\n";
      return -1;
    }

    out_file.write(reinterpret_cast<const char *>(
                       input_tensors_[model_name][i].sysMem.virAddr),
                   input_tensors_[model_name][i].sysMem.memSize);
    out_file.close();
    std::cout << "Saved " << filename << "\n";
  }
#endif
  hbDNNInferV3Param task_param;
  task_param.enable_pre_submit = infer_param.enable_pre_submit;
  task_param.enable_poll = false;

  LOGE_AND_RETURN_IF(
      hbDNNInferV3(&task_handle, output_tensors_[model_name].data(),
                   input_tensors_[model_name].data(),
                   *(dnn_handles_[packed_name_][model_name].get()),
                   &task_param),
      "hbDNNInferV3 for model[" + model_name + "] failed!", -1);
  return 0;
}

int32_t ModelManager::InferModel(std::string const &model_name,
                                 hbUCPTaskHandle_t &task_handle,
                                 int32_t task_id) {
// TODO(@horizon.ai):
// 当前多任务推理方式，仅支持单输入输出的模型，后续如有需求可对input_tensors_进行扩展。
#ifdef DUMP_MODEL_INPUT
  std::string filename =
      model_name + "_patch_" + std::to_string(task_id) + "_input_" + ".bin";
  std::ofstream out_file(filename, std::ios::out | std::ios::binary);
  if (!out_file) {
    std::cerr << "Error: Cannot open file " << filename << " for writing.\n";
    return -1;
  }

  out_file.write(reinterpret_cast<const char *>(
                     input_tensors_[model_name][i].sysMem.virAddr),
                 input_tensors_[model_name][i].sysMem.memSize);
  out_file.close();
  std::cout << "Saved " << filename << "\n";
#endif
  hbDNNTensor *input = &input_tensors_[model_name][task_id];
  hbDNNTensor *output = &output_tensors_[model_name][task_id];

  LOGE_AND_RETURN_IF(
      hbDNNInferV2(&task_handle, output, input,
                   *(dnn_handles_[packed_name_][model_name].get())),
      "hbDNNInferV2 for model[" + model_name + "] failed!", -1);
  return 0;
}

int32_t ModelManager::SubmitInferTask(std::string const &model_name,
                                      hbUCPTaskHandle_t &task_handle,
                                      InferParam const &infer_param) {
  hbUCPSchedParam ctrl_param;
  HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);

  if (infer_param.backend.empty()) {
    ctrl_param.backend = infer_backend_[InferBackend::kInferBackendBPUAny];
  } else {
    ctrl_param.backend = infer_backend_[infer_param.backend[0]];
    LOGD(TAG, "backend 0: {}", static_cast<int32_t>(infer_param.backend[0]));
    for (size_t i = 1; i < infer_param.backend.size(); ++i) {
      ctrl_param.backend |= infer_backend_[infer_param.backend[i]];
      LOGD(TAG, "backend {}: {}", i,
           static_cast<int32_t>(infer_param.backend[i]));
    }
  }

  ctrl_param.priority = infer_param.priority;
  ctrl_param.deviceId = infer_param.device_id;
  ctrl_param.customId = infer_param.custom_id;

  LOGE_AND_RETURN_IF(hbUCPSubmitTask(task_handle, &ctrl_param),
                     "hbUCPSubmitTask for model[" + model_name + "] failed!",
                     -1);

  LOGD(TAG, "Submit task success.");

  return 0;
}

int32_t ModelManager::NotifyTask(hbUCPTaskHandle_t &task_handle) {
  LOGE_AND_RETURN_IF(hbDNNNotifyTask(task_handle), "hbDNNNotifyTask failed!",
                     -1);
  return 0;
}

int32_t ModelManager::ReleaseTaskHandle(hbUCPTaskHandle_t &task_handle) {
  LOGE_AND_RETURN_IF(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed!",
                     -1);
  return 0;
}

void ModelManager::ReleaseTaskHandleAsnyc(hbUCPTaskHandle_t &task_handle) {
  TaskReleaseManager::GetInstance()->Post(task_handle);
}

int32_t ModelManager::ReleasePackedDnnHandle() {
  std::unique_lock<std::shared_mutex> packed_dnn_handle_lock(
      packed_dnn_handle_mutex_);

  auto packed_dnn_handle = packed_dnn_handle_.find(packed_name_);
  if (packed_dnn_handle != packed_dnn_handle_.end()) {
    auto pack_handle_use_count = pack_handle_use_count_.find(packed_name_);
    if (pack_handle_use_count != pack_handle_use_count_.end()) {
      pack_handle_use_count->second.fetch_sub(1);
      if (0 == pack_handle_use_count->second.load()) {
        hbDNNRelease(*packed_dnn_handle->second);
        packed_dnn_handle_.erase(packed_name_);
        pack_handle_use_count_.erase(packed_name_);
        dnn_handles_.erase(packed_name_);
      }
    }
  }

  return 0;
}

TaskReleaseManager::TaskReleaseManager() {
  for (int32_t i = 0; i < kReleaseTaskThreadNum; i++) {
    task_release_threads_.emplace_back(
        std::thread(&TaskReleaseManager::OnMsg, this));
  }
}

TaskReleaseManager::~TaskReleaseManager() {
  stop_ = true;
  cv_.notify_all();
  for (auto &thread : task_release_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  task_release_threads_.clear();
}

TaskReleaseManager *TaskReleaseManager::GetInstance() {
  static TaskReleaseManager instance;
  return &instance;
}

void TaskReleaseManager::Post(hbUCPTaskHandle_t task_handle) {
  {
    std::lock_guard<std::mutex> lk{mtx_};
    task_handles_.emplace(task_handle);
  }
  cv_.notify_one();
}

void TaskReleaseManager::OnMsg() {
  hbUCPTaskHandle_t task_handle = nullptr;
  while (!stop_) {
    {
      std::unique_lock<std::mutex> lk{mtx_};
      cv_.wait(lk, [&]() { return stop_ || !task_handles_.empty(); });
      if (stop_) {
        return;
      }
      task_handle = task_handles_.front();
      task_handles_.pop();
    }
    auto ret = hbUCPReleaseTask(task_handle);
    if (ret != 0) {
      LOGE(TAG, "hbUCPReleaseTask failed!");
    }
  }
}

}  // namespace xlm::utils
