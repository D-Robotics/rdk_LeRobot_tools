#include "pycauchyS100tools.h"
#define ALIGN(value, alignment) (((value) + ((alignment) - 1)) & ~((alignment) - 1))
#define ALIGN_32(value) ALIGN(value, 32)

bool checkFileExists(const std::string &path)
{
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

BPU_ACTPolicy::BPU_ACTPolicy(std::string &visionEncoder_model_path, std::string &transformerLayers_model_path)
{
    std::cout << " init BPU ACT Policy ..." << std::endl;
    // 检查 visionEncoder_model_path 是否存在且是文件
    if (!checkFileExists(visionEncoder_model_path))
    {
        throw std::runtime_error("Error: BPU ACT Policy VisionEncoder model path does not exist or is not a file: " + visionEncoder_model_path);
    }

    // 检查 transformerLayers_model_path 是否存在且是文件
    if (!checkFileExists(transformerLayers_model_path))
    {
        throw std::runtime_error("Error: BPU ACT Policy TransformerLayers model path does not exist or is not a file: " + transformerLayers_model_path);
    }

    // 加载 BPU ACT Policy VisionEncoder 模型
    auto visionEncoder_modelFileName = visionEncoder_model_path.c_str();
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&visionEncoder_packed_dnn_handle, &visionEncoder_modelFileName, 1),
        "BPU ACT Policy VisionEncoder hbDNNInitializeFromFiles failed");

    visionEncoder_model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&visionEncoder_model_name_list, &visionEncoder_model_count, visionEncoder_packed_dnn_handle),
        "BPU ACT Policy VisionEncoder hbDNNGetModelNameList failed");

    visionEncoder_model_name = visionEncoder_model_name_list[0];
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&visionEncoder_dnn_handle, visionEncoder_packed_dnn_handle, visionEncoder_model_name),
        "BPU ACT Policy VisionEncoder hbDNNGetModelHandle failed");

    // 模型输入检查
    visionEncoder_input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&visionEncoder_input_count, visionEncoder_dnn_handle),
        "BPU ACT Policy VisionEncoder hbDNNGetInputCount failed");
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&visionEncoder_input_properties, visionEncoder_dnn_handle, 0),
        "BPU ACT Policy VisionEncoder hbDNNGetInputTensorProperties failed");
    if (visionEncoder_input_count != 1)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input num is not 1, can't use this python api, please check.");
    if (visionEncoder_input_properties.tensorType != HB_DNN_TENSOR_TYPE_F32)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input type is not HB_DNN_TENSOR_TYPE_F32, can't use this python api, please check.");
    // if (visionEncoder_input_properties.tensorLayout != HB_DNN_LAYOUT_NCHW)
    // throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input layput is not HB_DNN_LAYOUT_NCHW, can't use this python api, please check.");
    if (visionEncoder_input_properties.validShape.numDimensions != 4)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input validShape.numDimensions is not 4, can't use this python api, please check.");
    if (visionEncoder_input_properties.validShape.dimensionSize[0] != 1 ||
        visionEncoder_input_properties.validShape.dimensionSize[1] != 3 ||
        visionEncoder_input_properties.validShape.dimensionSize[2] != 480 ||
        visionEncoder_input_properties.validShape.dimensionSize[3] != 640)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input shape is not (1,3,480,640), can't use this python api, please check.");

    // 模型输出检查
    visionEncoder_output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&visionEncoder_output_count, visionEncoder_dnn_handle),
        "BPU ACT Policy VisionEncoder hbDNNGetOutputCount failed");
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&visionEncoder_output_properties, visionEncoder_dnn_handle, 0),
        "BPU ACT Policy VisionEncoder hbDNNGetInputTensorProperties failed");
    if (visionEncoder_input_count != 1)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model output num is not 1, can't use this python api.");
    if (visionEncoder_output_properties.tensorType != HB_DNN_TENSOR_TYPE_F32)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input type is not HB_DNN_TENSOR_TYPE_F32, can't use this python api.");
    // if (visionEncoder_output_properties.tensorLayout != HB_DNN_LAYOUT_NCHW)
    //     throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input layput is not HB_DNN_LAYOUT_NCHW, can't use this python api.");
    if (visionEncoder_output_properties.validShape.numDimensions != 4)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input validShape.numDimensions is not 4, can't use this python api.");
    if (visionEncoder_output_properties.validShape.dimensionSize[0] != 1 ||
        visionEncoder_output_properties.validShape.dimensionSize[1] != 512 ||
        visionEncoder_output_properties.validShape.dimensionSize[2] != 15 ||
        visionEncoder_output_properties.validShape.dimensionSize[3] != 20)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model output shape is not (1,512,15,20), can't use this python api.");

    // laptop
    visionEncoder_laptop_input.resize(visionEncoder_input_count);
    for (int i = 0; i < visionEncoder_input_count; i++)
    {
        hbDNNGetInputTensorProperties(&visionEncoder_laptop_input[i].properties, visionEncoder_dnn_handle, i);
        RDK_CHECK_SUCCESS(
            hbUCPMallocCached(&visionEncoder_laptop_input[i].sysMem, visionEncoder_laptop_input[i].properties.alignedByteSize, 0),
            "BPU ACT Policy VisionEncoder laptop_input hbUCPMallocCached failed");
    }

    visionEncoder_laptop_output.resize(visionEncoder_output_count);
    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbDNNGetOutputTensorProperties(&visionEncoder_laptop_output[i].properties, visionEncoder_dnn_handle, i);
        RDK_CHECK_SUCCESS(
            hbUCPMallocCached(&visionEncoder_laptop_output[i].sysMem, visionEncoder_laptop_output[i].properties.alignedByteSize, 0),
            "BPU ACT Policy VisionEncoder laptop_output hbUCPMallocCached failed");
    }

    // phone
    visionEncoder_phone_input.resize(visionEncoder_input_count);
    for (int i = 0; i < visionEncoder_input_count; i++)
    {
        hbDNNGetInputTensorProperties(&visionEncoder_phone_input[i].properties, visionEncoder_dnn_handle, i);
        RDK_CHECK_SUCCESS(
            hbUCPMallocCached(&visionEncoder_phone_input[i].sysMem, visionEncoder_phone_input[i].properties.alignedByteSize, 0),
            "BPU ACT Policy VisionEncoder phone_input hbUCPMallocCached failed");
    }

    visionEncoder_phone_output.resize(visionEncoder_output_count);
    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbDNNGetOutputTensorProperties(&visionEncoder_phone_output[i].properties, visionEncoder_dnn_handle, i);
        RDK_CHECK_SUCCESS(
            hbUCPMallocCached(&visionEncoder_phone_output[i].sysMem, visionEncoder_phone_output[i].properties.alignedByteSize, 0),
            "BPU ACT Policy VisionEncoder phone_output hbUCPMallocCached failed");
    }

    // 加载 BPU ACT Policy TransformerLayers 模型
    auto transformerLayers_modelFileName = transformerLayers_model_path.c_str();
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&transformerLayers_packed_dnn_handle, &transformerLayers_modelFileName, 1),
        "BPU ACT Policy TransformerLayers hbDNNInitializeFromFiles failed");

    transformerLayers_model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&transformerLayers_model_name_list, &transformerLayers_model_count, transformerLayers_packed_dnn_handle),
        "BPU ACT Policy TransformerLayers hbDNNGetModelNameList failed");

    transformerLayers_model_name = transformerLayers_model_name_list[0];
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&transformerLayers_dnn_handle, transformerLayers_packed_dnn_handle, transformerLayers_model_name),
        "BPU ACT Policy TransformerLayers hbDNNGetModelHandle failed");

    // 模型输入检查
    transformerLayers_input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&transformerLayers_input_count, transformerLayers_dnn_handle),
        "BPU ACT Policy TransformerLayers hbDNNGetInputCount failed");
    if (transformerLayers_input_count != 3)
        throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model input num is not 3, can't use this python api, please check.");
    // RDK_CHECK_SUCCESS(
    //     hbDNNGetInputTensorProperties(&transformerLayers_input_properties, transformerLayers_dnn_handle, 0),
    //     "BPU ACT Policy TransformerLayers hbDNNGetInputTensorProperties failed");
    // if (transformerLayers_input_properties.tensorType != HB_DNN_TENSOR_TYPE_F32)
    //     throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model input type is not HB_DNN_TENSOR_TYPE_F32, can't use this python api, please check.");
    // // if (transformerLayers_input_properties.tensorLayout != HB_DNN_LAYOUT_NCHW)
    // // throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model input layput is not HB_DNN_LAYOUT_NCHW, can't use this python api, please check.");
    // if (transformerLayers_input_properties.validShape.numDimensions != 4)
    //     throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model input validShape.numDimensions is not 4, can't use this python api, please check.");
    // if (transformerLayers_input_properties.validShape.dimensionSize[0] != 1 ||
    //     transformerLayers_input_properties.validShape.dimensionSize[1] != 3 ||
    //     transformerLayers_input_properties.validShape.dimensionSize[2] != 480 ||
    //     transformerLayers_input_properties.validShape.dimensionSize[3] != 640)
    //     throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model input shape is not (1,3,480,640), can't use this python api, please check.");

    // 模型输出检查
    transformerLayers_output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&transformerLayers_output_count, transformerLayers_dnn_handle),
        "BPU ACT Policy TransformerLayers hbDNNGetOutputCount failed");
    if (transformerLayers_output_count != 1)
        throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model output num is not 1, can't use this python api.");
    // RDK_CHECK_SUCCESS(
    //     hbDNNGetOutputTensorProperties(&transformerLayers_output_properties, transformerLayers_dnn_handle, 0),
    //     "BPU ACT Policy TransformerLayers hbDNNGetInputTensorProperties failed");

    // if (transformerLayers_output_properties.tensorType != HB_DNN_TENSOR_TYPE_F32)
    //     throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model input type is not HB_DNN_TENSOR_TYPE_F32, can't use this python api.");
    // // if (transformerLayers_output_properties.tensorLayout != HB_DNN_LAYOUT_NCHW)
    // //     throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model input layput is not HB_DNN_LAYOUT_NCHW, can't use this python api.");
    // if (transformerLayers_output_properties.validShape.numDimensions != 3)
    //     throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model input validShape.numDimensions is not 3, can't use this python api.");
    // if (transformerLayers_output_properties.validShape.dimensionSize[0] != 1 ||
    //     transformerLayers_output_properties.validShape.dimensionSize[1] != 50 ||
    //     transformerLayers_output_properties.validShape.dimensionSize[2] != 6)
    //     throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model output shape is not (1,50,6), can't use this python api.");

    transformerLayers_input.resize(transformerLayers_input_count);
    for (int i = 0; i < transformerLayers_input_count; i++)
    {
        hbDNNGetInputTensorProperties(&transformerLayers_input[i].properties, transformerLayers_dnn_handle, i);
        if (i == 0)
        {
            RDK_CHECK_SUCCESS(
                hbUCPMallocCached(&transformerLayers_input[i].sysMem, transformerLayers_input[i].properties.alignedByteSize, 0),
                "BPU ACT Policy TransformerLayers laptop_input hbUCPMallocCached failed");
        }
        if (i == 1)
        {
            transformerLayers_input[i].sysMem = visionEncoder_laptop_output[0].sysMem;
        }
        if (i == 2)
        {
            transformerLayers_input[i].sysMem = visionEncoder_phone_output[0].sysMem;
        }
    }

    transformerLayers_output.resize(transformerLayers_output_count);
    for (int i = 0; i < transformerLayers_output_count; i++)
    {
        hbDNNGetOutputTensorProperties(&transformerLayers_output[i].properties, transformerLayers_dnn_handle, i);
        RDK_CHECK_SUCCESS(
            hbUCPMallocCached(&transformerLayers_output[i].sysMem, transformerLayers_output[i].properties.alignedByteSize, 0),
            "BPU ACT Policy TransformerLayers laptop_output hbUCPMallocCached failed");
    }

    // 打印输入输出信息
    {
        std::cout << "[BPU ACT Policy VisionEncoder]: " << visionEncoder_model_path << std::endl;
        std::cout << "[laptop]: " << std::endl;
        for (int32_t i = 0; i < visionEncoder_input_count; i++)
        {
            std::cout << "inputs[" << i << "]: (";
            for (int32_t j = 0; j < visionEncoder_laptop_input[i].properties.validShape.numDimensions; j++)
            {
                std::cout << visionEncoder_laptop_input[i].properties.validShape.dimensionSize[j] << ", ";
            }
            std::cout << ")" << std::endl;
        }
        for (int32_t i = 0; i < visionEncoder_output_count; i++)
        {
            std::cout << "outputs[" << i << "]: (";
            for (int32_t j = 0; j < visionEncoder_laptop_output[i].properties.validShape.numDimensions; j++)
            {
                std::cout << visionEncoder_laptop_output[i].properties.validShape.dimensionSize[j] << ", ";
            }
            std::cout << ")" << std::endl;
        }
        std::cout << "[phone]: " << std::endl;
        for (int32_t i = 0; i < visionEncoder_input_count; i++)
        {
            std::cout << "inputs[" << i << "]: (";
            for (int32_t j = 0; j < visionEncoder_phone_input[i].properties.validShape.numDimensions; j++)
            {
                std::cout << visionEncoder_phone_input[i].properties.validShape.dimensionSize[j] << ", ";
            }
            std::cout << ")" << std::endl;
        }
        for (int32_t i = 0; i < visionEncoder_output_count; i++)
        {
            std::cout << "outputs[" << i << "]: (";
            for (int32_t j = 0; j < visionEncoder_phone_output[i].properties.validShape.numDimensions; j++)
            {
                std::cout << visionEncoder_phone_output[i].properties.validShape.dimensionSize[j] << ", ";
            }
            std::cout << ")" << std::endl;
        }

        std::cout << "[BPU ACT Policy TransformerLayers]: " << transformerLayers_model_path << std::endl;
        for (int32_t i = 0; i < transformerLayers_input_count; i++)
        {
            std::cout << "inputs[" << i << "]: (";
            for (int32_t j = 0; j < transformerLayers_input[i].properties.validShape.numDimensions; j++)
            {
                std::cout << transformerLayers_input[i].properties.validShape.dimensionSize[j] << ", ";
            }
            std::cout << ")" << std::endl;
        }
        for (int32_t i = 0; i < transformerLayers_output_count; i++)
        {
            std::cout << "outputs[" << i << "]: (";
            for (int32_t j = 0; j < transformerLayers_output[i].properties.validShape.numDimensions; j++)
            {
                std::cout << transformerLayers_output[i].properties.validShape.dimensionSize[j] << ", ";
            }
            std::cout << ")" << std::endl;
        }
    }

    for (int32_t j = 0; j < transformerLayers_output[0].properties.validShape.numDimensions; j++)
    {
        shape.push_back
        (transformerLayers_output[0].properties.validShape.dimensionSize[j]);
    }
}

BPU_ACTPolicy::~BPU_ACTPolicy()
{

    // hbSysFreeMem(&(input.sysMem[0]));
    // for (int i = 0; i < output_count; i++)
    //     hbSysFreeMem(&(output[i].sysMem[0]));
    // hbDNNRelease(packed_dnn_handle);
    std::cout << "[INFO] release model success." << std::endl;
}

py::array_t<float> BPU_ACTPolicy::inference(py::array_t<float> state, py::array_t<float> laptop, py::array_t<float> phone)
{
    // np check
    py::buffer_info state_buf_info = state.request();
    if (state_buf_info.ndim != transformerLayers_input[0].properties.validShape.numDimensions ||
        state_buf_info.shape[0] != transformerLayers_input[0].properties.validShape.dimensionSize[0] ||
        state_buf_info.shape[1] != transformerLayers_input[0].properties.validShape.dimensionSize[1])
    {
        std::stringstream ss;
        ss << "wrong input numpy array state. need: (";
        for (int32_t i = 0; i < transformerLayers_input[0].properties.validShape.numDimensions; i++)
        {
            ss << transformerLayers_input[0].properties.validShape.dimensionSize[i] << ", ";
        }
        ss << "), got: (";
        for (int32_t i = 0; i < state_buf_info.ndim; i++)
        {
            ss << state_buf_info.shape[i] << ", ";
        }
        ss << ")";
        throw std::runtime_error(ss.str());
    }

    if (state_buf_info.format != py::format_descriptor<float>::format())
        throw std::runtime_error("Input numpy array state must have dtype float32.");

    py::buffer_info laptop_buf_info = laptop.request();
    if (laptop_buf_info.ndim != visionEncoder_laptop_input[0].properties.validShape.numDimensions ||
        laptop_buf_info.shape[0] != visionEncoder_laptop_input[0].properties.validShape.dimensionSize[0] ||
        laptop_buf_info.shape[1] != visionEncoder_laptop_input[0].properties.validShape.dimensionSize[1] ||
        laptop_buf_info.shape[2] != visionEncoder_laptop_input[0].properties.validShape.dimensionSize[2] ||
        laptop_buf_info.shape[3] != visionEncoder_laptop_input[0].properties.validShape.dimensionSize[3])
    {
        std::stringstream ss;
        ss << "wrong input numpy array laptop. need: (";
        for (int32_t i = 0; i < visionEncoder_laptop_input[0].properties.validShape.numDimensions; i++)
        {
            ss << visionEncoder_laptop_input[0].properties.validShape.dimensionSize[i] << ", ";
        }
        ss << "), got: (";
        for (int32_t i = 0; i < laptop_buf_info.ndim; i++)
        {
            ss << laptop_buf_info.shape[i] << ", ";
        }
        ss << ")";
        throw std::runtime_error(ss.str());
    }

    if (laptop_buf_info.format != py::format_descriptor<float>::format())
        throw std::runtime_error("Input numpy array laptop must have dtype float32.");

    py::buffer_info phone_buf_info = phone.request();
    if (phone_buf_info.ndim != visionEncoder_phone_input[0].properties.validShape.numDimensions ||
        phone_buf_info.shape[0] != visionEncoder_phone_input[0].properties.validShape.dimensionSize[0] ||
        phone_buf_info.shape[1] != visionEncoder_phone_input[0].properties.validShape.dimensionSize[1] ||
        phone_buf_info.shape[2] != visionEncoder_phone_input[0].properties.validShape.dimensionSize[2] ||
        phone_buf_info.shape[3] != visionEncoder_phone_input[0].properties.validShape.dimensionSize[3])
    {
        std::stringstream ss;
        ss << "wrong input numpy array phone. need: (";
        for (int32_t i = 0; i < visionEncoder_phone_input[0].properties.validShape.numDimensions; i++)
        {
            ss << visionEncoder_phone_input[0].properties.validShape.dimensionSize[i] << ", ";
        }
        ss << "), got: (";
        for (int32_t i = 0; i < laptop_buf_info.ndim; i++)
        {
            ss << phone_buf_info.shape[i] << ", ";
        }
        ss << ")";
        throw std::runtime_error(ss.str());
    }

    if (phone_buf_info.format != py::format_descriptor<float>::format())
        throw std::runtime_error("Input numpy array phone must have dtype float32.");

    // 启动 laptop 的 VisonEncoder 特征推理
    float *laptop_np_ptr = reinterpret_cast<float *>(laptop_buf_info.ptr);
    float *laptop_hbTensor_ptr = reinterpret_cast<float *>(visionEncoder_laptop_input[0].sysMem.virAddr);
    std::memcpy(laptop_hbTensor_ptr, laptop_np_ptr, laptop_buf_info.size * sizeof(float));

    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbUCPMemFlush(&visionEncoder_laptop_input[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    hbUCPTaskHandle_t laptop_task_handle{nullptr};

    RDK_CHECK_SUCCESS(
        hbDNNInferV2(&laptop_task_handle, visionEncoder_laptop_output.data(), visionEncoder_laptop_input.data(), visionEncoder_dnn_handle),
        "BPU ACT Policy Vision Encoder laptop_task hbDNNInferV2 failed");
    hbUCPSchedParam laptop_ctrl_param;

    HB_UCP_INITIALIZE_SCHED_PARAM(&laptop_ctrl_param);
    laptop_ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    RDK_CHECK_SUCCESS(hbUCPSubmitTask(laptop_task_handle, &laptop_ctrl_param),
                      "BPU ACT Policy Vision Encoder laptop_task hbUCPSubmitTask failed");

    // 启动 phone 的 VisonEncoder 特征推理
    float *phone_np_ptr = reinterpret_cast<float *>(phone_buf_info.ptr);
    float *phone_hbTensor_ptr = reinterpret_cast<float *>(visionEncoder_phone_input[0].sysMem.virAddr);
    std::memcpy(phone_hbTensor_ptr, phone_np_ptr, phone_buf_info.size * sizeof(float));

    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbUCPMemFlush(&visionEncoder_phone_input[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    hbUCPTaskHandle_t phone_task_handle{nullptr};

    RDK_CHECK_SUCCESS(
        hbDNNInferV2(&phone_task_handle, visionEncoder_phone_output.data(), visionEncoder_phone_input.data(), visionEncoder_dnn_handle),
        "BPU ACT Policy Vision Encoder phone_task hbDNNInferV2 failed");
    hbUCPSchedParam phone_ctrl_param;

    HB_UCP_INITIALIZE_SCHED_PARAM(&phone_ctrl_param);
    phone_ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    RDK_CHECK_SUCCESS(hbUCPSubmitTask(phone_task_handle, &phone_ctrl_param),
                      "BPU ACT Policy Vision Encoder phone_task hbUCPSubmitTask failed");

    // 为 TransformerLayers 准备输入数据
    float *state_np_ptr = reinterpret_cast<float *>(state_buf_info.ptr);
    float *state_hbTensor_ptr = reinterpret_cast<float *>(transformerLayers_input[0].sysMem.virAddr);
    std::memcpy(state_hbTensor_ptr, state_np_ptr, state_buf_info.size * sizeof(float));

    // 等待 laptop 的推理结束
    RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(laptop_task_handle, 0),
                      "BPU ACT Policy Vision Encoder laptop_task hbUCPWaitTaskDone failed");

    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbUCPMemFlush(&visionEncoder_laptop_output[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // laptop_hbTensor_ptr = reinterpret_cast<float *>(visionEncoder_laptop_output[0].sysMem.virAddr);
    RDK_CHECK_SUCCESS(hbUCPReleaseTask(laptop_task_handle), "BPU ACT Policy Vision Encoder laptop_task hbUCPReleaseTask failed");

    // 等待 phone 的推理结束
    RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(phone_task_handle, 0),
                      "BPU ACT Policy Vision Encoder phone_task hbUCPWaitTaskDone failed");

    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbUCPMemFlush(&visionEncoder_phone_output[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // phone_hbTensor_ptr = reinterpret_cast<float *>(visionEncoder_phone_output[0].sysMem.virAddr);
    RDK_CHECK_SUCCESS(hbUCPReleaseTask(phone_task_handle), "BPU ACT Policy Vision Encoder phone_task hbUCPReleaseTask failed");

    // 启动 TransformerLayers 的推理
    for (int i = 0; i < transformerLayers_input_count; i++)
    {
        hbUCPMemFlush(&transformerLayers_input[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    hbUCPTaskHandle_t transformerLayers_task_handle{nullptr};
    RDK_CHECK_SUCCESS(
        hbDNNInferV2(&transformerLayers_task_handle, transformerLayers_output.data(), transformerLayers_input.data(), transformerLayers_dnn_handle),
        "BPU ACT Policy TransformerLayers phone_task hbDNNInferV2 failed");

    hbUCPSchedParam transformerLayers_ctrl_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&transformerLayers_ctrl_param);
    phone_ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    RDK_CHECK_SUCCESS(hbUCPSubmitTask(transformerLayers_task_handle, &transformerLayers_ctrl_param),
                      "BPU ACT Policy TransformerLayers hbUCPSubmitTask failed");

    // 等待 TransformerLayers 推理结束
    RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(transformerLayers_task_handle, 0),
                      "BPU ACT Policy TransformerLayers hbUCPWaitTaskDone failed");

    for (int i = 0; i < transformerLayers_output_count; i++)
    {
        hbUCPMemFlush(&transformerLayers_output[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    auto actions_hbTensor_ptr = reinterpret_cast<float *>(transformerLayers_output[0].sysMem.virAddr);
    RDK_CHECK_SUCCESS(hbUCPReleaseTask(transformerLayers_task_handle), "BPU ACT Policy TransformerLayers hbUCPReleaseTask failed");

    // 返回多维 NumPy 数组

    return py::array_t<float>(shape, actions_hbTensor_ptr);
}
