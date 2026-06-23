#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"
#include "hobot/hb_ucp_status.h"

namespace py = pybind11;

#define BPU_ALIGN(value, align) (((value) + ((align)-1)) & ~((align)-1))

class BPUSubModel {
 public:
    BPUSubModel(const std::string& model_path) : model_path_(model_path) {
        const char* path = model_path.c_str();
        check_success(hbDNNInitializeFromFiles(&packed_handle_, &path, 1),
                      "hbDNNInitializeFromFiles failed for " + model_path);

        const char** name_list;
        check_success(hbDNNGetModelNameList(&name_list, &model_count_, packed_handle_),
                      "hbDNNGetModelNameList failed");
        check_success(hbDNNGetModelHandle(&dnn_handle_, packed_handle_, name_list[0]),
                      "hbDNNGetModelHandle failed");

        check_success(hbDNNGetInputCount(&input_count_, dnn_handle_), "GetInputCount");
        check_success(hbDNNGetOutputCount(&output_count_, dnn_handle_), "GetOutputCount");

        input_tensors_.resize(input_count_);
        output_tensors_.resize(output_count_);

        for (int i = 0; i < input_count_; ++i) {
            check_success(hbDNNGetInputTensorProperties(&input_tensors_[i].properties, dnn_handle_, i),
                          "GetInputTensorProperties");
            const char* iname = nullptr;
            hbDNNGetInputName(&iname, dnn_handle_, i);
            if (iname) input_names_[std::string(iname)] = i;
        }
        for (int i = 0; i < output_count_; ++i)
            check_success(hbDNNGetOutputTensorProperties(&output_tensors_[i].properties, dnn_handle_, i),
                          "GetOutputTensorProperties");

        const int align = 32;
        for (int i = 0; i < input_count_; ++i) {
            auto& t = input_tensors_[i];
            auto ndim = t.properties.validShape.numDimensions;
            for (int d = ndim - 1; d >= 0; --d) {
                if (t.properties.stride[d] == -1) {
                    auto s = t.properties.stride[d + 1] * t.properties.validShape.dimensionSize[d + 1];
                    t.properties.stride[d] = BPU_ALIGN(s, align);
                }
            }
            int sz = t.properties.stride[0] * t.properties.validShape.dimensionSize[0];
            check_success(hbUCPMallocCached(&t.sysMem, sz, 0), "MallocCached input");
        }
        for (int i = 0; i < output_count_; ++i) {
            int sz = output_tensors_[i].properties.alignedByteSize;
            check_success(hbUCPMallocCached(&output_tensors_[i].sysMem, sz, 0), "MallocCached output");
        }
    }

    ~BPUSubModel() {
        for (int i = 0; i < input_count_; ++i) hbUCPFree(&input_tensors_[i].sysMem);
        for (int i = 0; i < output_count_; ++i) hbUCPFree(&output_tensors_[i].sysMem);
        hbDNNRelease(packed_handle_);
    }

    py::dict run(const std::map<std::string, py::array_t<float>>& inputs) {
        if (static_cast<int>(inputs.size()) != input_count_) {
            throw std::runtime_error("Expected " + std::to_string(input_count_) +
                                     " inputs, got " + std::to_string(inputs.size()));
        }

        for (auto& [name, arr] : inputs) {
            auto it = input_names_.find(name);
            int idx;
            if (it != input_names_.end()) {
                idx = it->second;
            } else {
                idx = 0;
                for (auto& [n, i] : input_names_) {
                    if (n.find(name) != std::string::npos || name.find(n) != std::string::npos) {
                        idx = i;
                        break;
                    }
                }
            }
            write_tensor(input_tensors_[idx], arr);
        }

        infer();

        py::dict result;
        for (int i = 0; i < output_count_; ++i) {
            result[py::str(get_output_name(i))] = read_tensor(output_tensors_[i]);
        }
        return result;
    }

 private:
    void infer() {
        hbUCPTaskHandle_t task{nullptr};
        check_success(hbDNNInferV2(&task, output_tensors_.data(), input_tensors_.data(), dnn_handle_),
                      "hbDNNInferV2 failed");
        hbUCPSchedParam param;
        HB_UCP_INITIALIZE_SCHED_PARAM(&param);
        param.backend = HB_UCP_BPU_CORE_ANY;
        check_success(hbUCPSubmitTask(task, &param), "hbUCPSubmitTask");
        check_success(hbUCPWaitTaskDone(task, 0), "hbUCPWaitTaskDone");
        for (int i = 0; i < output_count_; ++i)
            hbUCPMemFlush(&output_tensors_[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
        check_success(hbUCPReleaseTask(task), "hbUCPReleaseTask");
    }

    void write_tensor(hbDNNTensor& tensor, const py::array_t<float>& arr) {
        auto buf = arr.request();
        auto& props = tensor.properties;
        auto& shape = props.validShape;
        uint8_t* base = reinterpret_cast<uint8_t*>(tensor.sysMem.virAddr);
        const int64_t* stride = props.stride;
        int ndim = shape.numDimensions;

        if (ndim == 4) {
            int N = shape.dimensionSize[0], C = shape.dimensionSize[1];
            int H = shape.dimensionSize[2], W = shape.dimensionSize[3];
            const float* src = static_cast<float*>(buf.ptr);
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c)
                    for (int h = 0; h < H; ++h) {
                        float* dst = reinterpret_cast<float*>(
                            base + n * stride[0] + c * stride[1] + h * stride[2]);
                        memcpy(dst, src + ((n * C + c) * H + h) * W, W * sizeof(float));
                    }
        } else if (ndim == 2) {
            int N = shape.dimensionSize[0], D = shape.dimensionSize[1];
            const float* src = static_cast<float*>(buf.ptr);
            for (int n = 0; n < N; ++n) {
                float* dst = reinterpret_cast<float*>(base + n * stride[0]);
                memcpy(dst, src + n * D, D * sizeof(float));
            }
        } else {
            int64_t total = 1;
            for (int i = 0; i < ndim; ++i) total *= shape.dimensionSize[i];
            memcpy(base, buf.ptr, total * sizeof(float));
        }
        hbUCPMemFlush(&tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }

    py::array read_tensor(const hbDNNTensor& tensor) {
        auto& props = tensor.properties;
        auto& shape = props.validShape;
        const int64_t* stride = props.stride;
        int ndim = shape.numDimensions;

        std::vector<ssize_t> dims(ndim);
        int64_t total = 1;
        for (int i = 0; i < ndim; ++i) {
            dims[i] = shape.dimensionSize[i];
            total *= dims[i];
        }

        py::array_t<float> result(dims);
        auto buf = result.request();
        float* out = static_cast<float*>(buf.ptr);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(tensor.sysMem.virAddr);

        if (props.tensorType == HB_DNN_TENSOR_TYPE_F32) {
            if (ndim == 4) {
                int N = dims[0], C = dims[1], H = dims[2], W = dims[3];
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C; ++c)
                        for (int h = 0; h < H; ++h) {
                            const float* src = reinterpret_cast<const float*>(
                                base + n * stride[0] + c * stride[1] + h * stride[2]);
                            memcpy(out + ((n * C + c) * H + h) * W, src, W * sizeof(float));
                        }
            } else if (ndim == 3) {
                int N = dims[0], S = dims[1], D = dims[2];
                for (int n = 0; n < N; ++n)
                    for (int s = 0; s < S; ++s) {
                        const float* src = reinterpret_cast<const float*>(
                            base + n * stride[0] + s * stride[1]);
                        memcpy(out + (n * S + s) * D, src, D * sizeof(float));
                    }
            } else {
                memcpy(out, base, total * sizeof(float));
            }
        } else if (props.tensorType == HB_DNN_TENSOR_TYPE_S16 || props.tensorType == HB_DNN_TENSOR_TYPE_F16) {
            float scale_val = 1.0f;
            if (props.scale.scaleData && props.scale.scaleLen > 0)
                scale_val = props.scale.scaleData[0];

            if (ndim == 4) {
                int N = dims[0], C = dims[1], H = dims[2], W = dims[3];
                for (int n = 0; n < N; ++n)
                    for (int c = 0; c < C; ++c) {
                        int si = (props.quantizeAxis == 1 && props.scale.scaleLen > 1) ? c : 0;
                        float s = props.scale.scaleData ? props.scale.scaleData[si] : 1.0f;
                        for (int h = 0; h < H; ++h) {
                            const int16_t* src = reinterpret_cast<const int16_t*>(
                                base + n * stride[0] + c * stride[1] + h * stride[2]);
                            float* dst = out + ((n * C + c) * H + h) * W;
                            for (int w = 0; w < W; ++w) dst[w] = src[w] * s;
                        }
                    }
            } else if (ndim == 3) {
                int N = dims[0], S = dims[1], D = dims[2];
                for (int n = 0; n < N; ++n)
                    for (int s = 0; s < S; ++s) {
                        const int16_t* src = reinterpret_cast<const int16_t*>(
                            base + n * stride[0] + s * stride[1]);
                        float* dst = out + (n * S + s) * D;
                        for (int d = 0; d < D; ++d) dst[d] = src[d] * scale_val;
                    }
            } else {
                const int16_t* src = reinterpret_cast<const int16_t*>(base);
                for (int64_t i = 0; i < total; ++i) out[i] = src[i] * scale_val;
            }
        } else if (props.tensorType == HB_DNN_TENSOR_TYPE_S8 || props.tensorType == HB_DNN_TENSOR_TYPE_U8) {
            float scale_val = 1.0f;
            if (props.scale.scaleData && props.scale.scaleLen > 0)
                scale_val = props.scale.scaleData[0];
            bool is_signed = (props.tensorType == HB_DNN_TENSOR_TYPE_S8);
            for (int64_t i = 0; i < total; ++i) {
                float v = is_signed ? static_cast<float>(*reinterpret_cast<const int8_t*>(base + i))
                                    : static_cast<float>(base[i]);
                out[i] = v * scale_val;
            }
        } else {
            memcpy(out, base, total * sizeof(float));
        }

        return result;
    }

    std::string get_output_name(int idx) {
        const char* name = nullptr;
        int32_t err = hbDNNGetOutputName(&name, dnn_handle_, idx);
        if (err == 0 && name) return std::string(name);
        return "output_" + std::to_string(idx);
    }

    void check_success(int32_t err, const std::string& ctx) {
        if (err != 0) {
            const char* desc = hbUCPGetErrorDesc(err);
            throw std::runtime_error("BPU Error (code: " + std::to_string(err) +
                                     ", desc: " + (desc ? desc : "?") + ") " + ctx);
        }
    }

    std::string model_path_;
    hbDNNPackedHandle_t packed_handle_{nullptr};
    hbDNNHandle_t dnn_handle_{nullptr};
    int model_count_ = 0;
    int32_t input_count_ = 0;
    int32_t output_count_ = 0;
    std::vector<hbDNNTensor> input_tensors_;
    std::vector<hbDNNTensor> output_tensors_;
    std::map<std::string, int> input_names_;
};

class BPUACTRuntime {
 public:
    BPUACTRuntime(const std::vector<std::string>& model_paths) {
        for (const auto& p : model_paths) {
            models_.push_back(std::make_unique<BPUSubModel>(p));
        }
    }

    py::dict run(const std::map<std::string, py::array_t<float>>& inputs,
                 const std::string& model_name) {
        if (model_name.find("VisionEncoder") != std::string::npos && models_.size() >= 1) {
            return models_[0]->run(inputs);
        } else if (model_name.find("Transformer") != std::string::npos && models_.size() >= 2) {
            return models_[1]->run(inputs);
        } else if (models_.size() == 1) {
            return models_[0]->run(inputs);
        }
        throw std::runtime_error("Unknown model_name: " + model_name);
    }

 private:
    std::vector<std::unique_ptr<BPUSubModel>> models_;
};
