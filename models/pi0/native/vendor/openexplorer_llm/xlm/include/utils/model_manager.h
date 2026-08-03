/*
 * Copyright (C) 2024 Shanghai Gua Technology Co., Ltd.
 * All rights reserved
 */
#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

namespace xlm::utils {

enum class FlushType {
  kFlushInvalidate = 0,  // 将内存同步到缓存中，CPU读前使用
  kFlushClean = 1,       // 将缓存数据同步到内存中，CPU写后使用
};

enum class InferBackend {
  kInferBackendCoreAny = 0,  // 任意核
  kInferBackendBPUAny = 1,   // 任意BPU核
  kInferBackendBPU0 = 2,     // BPU 0
  kInferBackendBPU1 = 3,     // BPU 1
  kInferBackendBPU2 = 4,     // BPU 2
  kInferBackendBPU3 = 5,     // BPU 3
};

struct InferParam {
  int32_t priority{
      HB_UCP_PRIORITY_LOWEST};  // 推理优先级[0~255]，值越小优先级越高
  int32_t device_id{0};         //
  int32_t custom_id{0};         //
  mutable std::vector<InferBackend> backend{};  // 推理后端
  bool enable_pre_submit{false};                // 支持提前下发
};

enum class MemAllocType {
  kMemAllocUnique = 0,        // 独占内存
  kMemAllocSharedInput = 1,   // 共享input tensor内存
  kMemAllocSharedOutput = 2,  // 共享output tensor内存
  kMemAllocSharedBoth = 3,    // 共享input与output tensor内存
};

class ModelManager;

class TaskReleaseManager {
 public:
  static TaskReleaseManager *GetInstance();
  void Post(hbUCPTaskHandle_t task_handle);

 private:
  TaskReleaseManager();
  ~TaskReleaseManager();
  void OnMsg();

 private:
  std::vector<std::thread> task_release_threads_{};
  bool stop_ = false;
  std::mutex mtx_{};
  std::condition_variable cv_{};
  std::queue<hbUCPTaskHandle_t> task_handles_{};
};

/**
 * @brief
 * tensor内存申请函数类型
 *
 * @param ModelManager* ModelManager指针
 * @return int 0 success other failed
 */
using AllocTensorMemFunc = std::function<int32_t(ModelManager *)>;

/**
 * @brief
 * tensor内存释放函数类型
 *
 * @param ModelManager* ModelManager指针
 * @return int 0 success other failed
 */
using FreeTensorMemFunc = std::function<int32_t(ModelManager *)>;

struct ModelInfo {
  std::string hbm_name;  // hbm模型名称，可由用户自定义
  std::string hbm_path;  // hbm模型所在路径
  int32_t device_id{0};  // 预留参数，ucp hbUCPMallocCached接口需要
  int32_t input_mem_multi{1};   // input tensor 内存申请倍数
  int32_t output_mem_multi{1};  // output tensor 内存申请倍数
  int32_t max_parallel_cnt{1};  // 模型并行处理数据的最大数量
  MemAllocType mem_alloc_type{
      MemAllocType::kMemAllocUnique};  // 内存申请模式，详见 MemAllocType
  int32_t shared_input_start_index{
      3};  // input tensor共享内存的起始index,仅申请共享内存是需要
  int32_t shared_output_start_index{
      0};  // output tensor共享内存的起始index,仅申请共享内存是需要
  int32_t kv_mem_multi{3};         // kv cache内存申请倍数
  int32_t non_kv_input_count{3};   // 模型输入中非kv cache的tensor个数
  int32_t non_kv_output_count{1};  // 模型输出中非kv cache的tensor个数
  bool use_llm_model{false};       // 当前模型是否为llm模型
  bool use_draft_model{false};     // 当前模型是否为draft模型
  AllocTensorMemFunc alloc_tensor_mem_func{
      nullptr};  // tensor内存申请函数，支持用户自定义内存方法
  FreeTensorMemFunc free_tensor_mem_func{
      nullptr};  // tensor内存释放函数，支持用户自定义内存方法
};

/**
 * @brief
 * 异步推理任务回调函数
 *
 * @param hbUCPTaskHandle_t task 句柄
 * @param int32_t status 任务返回的状态码
 * @param string model name
 * @return int 0 success other failed
 */
using TaskDoneCb = std::function<void(hbUCPTaskHandle_t, int32_t,
                                      std::string const)>;  // NOLINT

struct PreSubmitParam {
  std::function<int32_t(void)> update_func{
      nullptr};                // 用于提前下发任务更新地址信息
  std::string dnn_model_name;  // 模型名称
  bool need_wait{false};       // 是否需要等待任务完成
  std::promise<int> infer_result;
  TaskDoneCb cb{nullptr};  // 任务完成回调函数指针
};

class ModelManager {
 public:
  // explicit ModelManager() = default;
  ModelManager();

  ~ModelManager();

  /**
   * @brief
   * 初始化接口
   *
   * @param model_info 加载模型信息，详见ModelInfo结构体
   * @return int 返回值
   */
  [[nodiscard]] int32_t Init(ModelInfo const &model_info);

  /**
   * @brief
   * 去初始化接口，可缺省，在析构函数中会自动调用
   *
   * @return int 返回值
   */
  int32_t DeInit();

  int32_t DeInitShared();

  /**
   * @brief
   * 获取input tensors
   *
   * @param model_name 模型名称
   * @param input_tensors 返回获取到的input_tensors
   * @return int 0 success other failed
   */
  [[nodiscard]] int32_t GetInputTensors(
      std::string const &model_name, std::vector<hbDNNTensor> *&input_tensors);

  /**
   * @brief
   * 获取output tensors
   *
   * @param model_name 模型名称
   * @param input_tensors 返回获取到的output_tensors
   * @return int 0 success other failed
   */
  [[nodiscard]] int32_t GetOutputTensors(
      std::string const &model_name, std::vector<hbDNNTensor> *&output_tensors);

  /**
   * @brief
   * 刷新同步输入tensor的内存数据
   *
   * @param model_name 模型名称
   * @param flush_type 内存数据刷新类型，详见FlushType
   * @param flush_tensor_ids 需要刷新的tensor id
   * @return int 0 success other failed
   */
  [[nodiscard]] int32_t FlushInputMem(
      std::string const &model_name, FlushType const &flush_type,
      std::vector<int32_t> const &flush_tensor_ids = {});

  /**
   * @brief
   * 刷新同步输出tensor的内存数据
   *
   * @param model_name 模型名称
   * @param flush_type 内存数据刷新类型，详见FlushType
   * @param flush_tensor_ids 需要刷新的tensor id
   * @return int 0 success other failed
   */
  [[nodiscard]] int32_t FlushOutputMem(
      std::string const &model_name, FlushType const &flush_type,
      std::vector<int32_t> const &flush_tensor_ids = {});

  /**
   * @brief
   * 提交同步推理任务
   *
   * @param model_name 模型名称
   * @param task_handle 任务句柄
   * @param infer_param 推理参数设置，详见InferParam
   * @param timeout 同步推理任务超时时间
   * @return int 0 success other failed
   */
  [[nodiscard]] int32_t InferTaskSync(std::string const &model_name,
                                      hbUCPTaskHandle_t &task_handle,
                                      InferParam const &infer_param,
                                      int32_t const timeout = 0,
                                      int32_t const task_id = -1);

  /**
   * @brief
   * 提交异步推理任务
   *
   * @param model_name 模型名称
   * @param task_handle 任务句柄
   * @param infer_param 推理参数设置，详见InferParam
   * @param task_done_cb 异步推理任务结果回调函数
   * @return int 0 success other failed
   */
  [[nodiscard]] int32_t InferTaskAsync(std::string const &model_name,
                                       hbUCPTaskHandle_t &task_handle,
                                       InferParam const &infer_param,
                                       TaskDoneCb task_done_cb);

  /**
   * @brief
   * 提交提前下发异步推理任务
   *
   * @param infer_param 推理参数设置，详见InferParam
   * @param pre_submit_param 提前下发参数设置，详见 PreSubmitParam
   * @param task_handle 任务句柄
   * @return int 0 success other failed
   */
  int32_t InferPreSubmitTask(InferParam const &infer_param,
                             PreSubmitParam &pre_submit_param,
                             hbUCPTaskHandle_t &task_handle);

  /**
   * @brief
   * 唤醒提前下任务
   *
   * @param task_handle 任务句柄
   * @return int 0 success other failed
   */
  int32_t NotifyTask(hbUCPTaskHandle_t &task_handle);

  /**
   * @brief
   * 释放task句柄
   *
   * @param task_handle 任务句柄
   * @return int 0 success other failed
   */
  int32_t ReleaseTaskHandle(hbUCPTaskHandle_t &task_handle);

  /**
   * @brief
   * 异步释放task句柄，交由一个线程完成
   *
   * @param task_handle 任务句柄
   * @return
   */
  void ReleaseTaskHandleAsnyc(hbUCPTaskHandle_t &task_handle);

 public:
  inline int32_t const model_count() const noexcept { return model_count_; }

  inline ModelInfo const &model_info() const noexcept { return model_info_; }

  /**
   * @brief 读取已缓存的某个子模型编译BPU核数
   * @note 该缓存在ModelManager::Init()初始化时自动填充
   *
   * @param model_name 子模型名称
   * @return int32_t BPU核数；未找到返回-1
   */
  inline int32_t const bpu_core_num(
      std::string const &model_name) const noexcept {
    if (bpu_core_num_.find(model_name) == bpu_core_num_.end()) {
      return -1;
    }

    return bpu_core_num_.at(model_name);
  }

  /**
   * @brief 获取所有子模型编译BPU核数的映射表
   * @note 该缓存在ModelManager::Init()初始化时自动填充
   *
   * @return 模型名称到BPU核数的映射
   */
  inline std::unordered_map<std::string, int32_t> const &bpu_core_num_map()
      const noexcept {
    return bpu_core_num_;
  }

  inline std::unordered_map<
      std::string,
      std::unordered_map<std::string, std::shared_ptr<hbDNNHandle_t>>> const &
  dnn_handles() const noexcept {
    return dnn_handles_;
  }

  inline std::string const &packed_name() const noexcept {
    return packed_name_;
  }

  inline std::vector<std::string> const &model_names() const noexcept {
    return model_names_;
  }

  inline std::unordered_map<std::string, int32_t> const &input_counts()
      const noexcept {
    return input_counts_;
  }

  inline std::unordered_map<std::string, int32_t> const &output_counts()
      const noexcept {
    return output_counts_;
  }

  inline hbUCPSysMem &mutable_inoutput_mem() noexcept {
    return model_inout_mem_;
  }

  inline std::unordered_map<std::string, std::vector<hbDNNTensor>> &
  mutable_input_tensors() noexcept {
    return input_tensors_;
  }

  inline std::unordered_map<std::string, std::vector<hbDNNTensor>> &
  mutable_output_tensors() noexcept {
    return output_tensors_;
  }

 protected:
  static void HbUcpTaskDoneCallBack(hbUCPTaskHandle_t taskHandle,
                                    int32_t status, void *userdata);

  int32_t LoadHbmPack(std::string const &hbm_name, std::string const &hbm_path);

  int32_t GetModelNames(std::string const &hbm_name);

  int32_t GetBpuCoreNum(std::string const &model_name);

  int32_t LoadDnnModel(std::string const &hbm_name);

  int32_t GetInOutputCounts();

  int32_t DefaultAllocTensorMem();

  int32_t AllocUniqueTensorMem(ModelInfo const &model_info,
                               std::string const &model_name);

  int32_t InitInOutputTensorsShared(ModelInfo const &model_info);

  int32_t AllocSharedInputTensorMem(
      ModelInfo const &model_info, std::vector<std::string> const &model_names);

  int32_t InferModel(std::string const &model_name,
                     hbUCPTaskHandle_t &task_handle,
                     InferParam const &infer_param);

  int32_t InferModel(std::string const &model_name,
                     hbUCPTaskHandle_t &task_handle, int32_t task_id);

  int32_t SubmitInferTask(std::string const &model_name,
                          hbUCPTaskHandle_t &task_handle,
                          InferParam const &infer_param);

  int32_t FlushMem(std::string const &model_name, FlushType const &flush_type,
                   std::vector<int32_t> const &flush_tensor_ids,
                   std::unordered_map<std::string, std::vector<hbDNNTensor>>
                       &tensor_storage);

  int32_t GetTensors(std::string const &model_name,
                     std::vector<hbDNNTensor> *&tensors,
                     std::unordered_map<std::string, std::vector<hbDNNTensor>>
                         &tensor_storage);

  int32_t ReleasePackedDnnHandle();

 protected:
  // 设置为static变量，以实现对于同一个模型的不同实例只加在一次模型
  // key: packed model name
  // value: packed dnn handle
  static std::unordered_map<std::string, std::unique_ptr<hbDNNPackedHandle_t>>
      packed_dnn_handle_;
  // key: packed model name
  // value: dnn_handles
  //          key: model name
  //          value: dnn handle
  static std::unordered_map<
      std::string,
      std::unordered_map<std::string, std::shared_ptr<hbDNNHandle_t>>>
      dnn_handles_;

  static std::unordered_map<std::string, std::atomic<int32_t>>
      pack_handle_use_count_;

  // 对于不同实例在不同线程里访问packed_dnn_handle_和dnn_handles_时加锁保护
  static std::shared_mutex packed_dnn_handle_mutex_;
  static std::shared_mutex dnn_handles_mutex_;
  static std::shared_mutex task_done_callback_mutex_;

  // 以下参数为每个实例独有
  TaskDoneCb task_done_callback_;
  std::string infer_model_name_;
  int32_t model_count_{0};
  std::vector<std::string> model_names_{};

  std::string packed_name_;
  ModelInfo model_info_;

  // 以下map key均为 model name
  std::unordered_map<std::string, int32_t> input_counts_;
  std::unordered_map<std::string, int32_t> output_counts_;

  std::unordered_map<std::string, int32_t> bpu_core_num_;

  std::unordered_map<std::string, std::vector<hbDNNTensor>> input_tensors_;
  std::unordered_map<std::string, std::vector<hbDNNTensor>> output_tensors_;

  std::unordered_map<FlushType, hbUCPSysMemFlushFlag> hcup_flush_flags_;
  std::unordered_map<InferBackend, int32_t> infer_backend_;

  // 判断是否是特殊模型 走不同的析构函数
  bool is_deepseek_model_ = false;
  hbUCPSysMem model_inout_mem_ = hbUCPSysMem{0U, nullptr, 0U};
  bool is_omni_text_model_ = false;
  bool is_qwen_model_ = false;
};
}  // namespace xlm::utils
