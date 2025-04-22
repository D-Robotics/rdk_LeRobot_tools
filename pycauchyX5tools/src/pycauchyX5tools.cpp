#include "pycauchyX5tools.h"
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
    if (visionEncoder_input_properties.tensorLayout != HB_DNN_LAYOUT_NCHW)
    throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input layput is not HB_DNN_LAYOUT_NCHW, can't use this python api, please check.");
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
    if (visionEncoder_output_properties.tensorLayout != HB_DNN_LAYOUT_NCHW)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input layput is not HB_DNN_LAYOUT_NCHW, can't use this python api.");
    if (visionEncoder_output_properties.validShape.numDimensions != 4)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model input validShape.numDimensions is not 4, can't use this python api.");
    if (visionEncoder_output_properties.validShape.dimensionSize[0] != 1 ||
        visionEncoder_output_properties.validShape.dimensionSize[1] != 512 ||
        visionEncoder_output_properties.validShape.dimensionSize[2] != 15 ||
        visionEncoder_output_properties.validShape.dimensionSize[3] != 20)
        throw std::runtime_error("Your BPU ACT Policy VisionEncoder Model output shape is not (1,512,15,20), can't use this python api.");

    // laptop
    visionEncoder_laptop_input = new hbDNNTensor[visionEncoder_input_count];
    for (int i = 0; i < visionEncoder_input_count; i++)
    {
        hbDNNGetInputTensorProperties(&visionEncoder_laptop_input[i].properties, visionEncoder_dnn_handle, i);
        RDK_CHECK_SUCCESS(
            hbSysAllocCachedMem(&visionEncoder_laptop_input[i].sysMem[0], visionEncoder_laptop_input[i].properties.alignedByteSize),
            "BPU ACT Policy VisionEncoder laptop_input hbSysAllocCachedMem failed");
    }

    visionEncoder_laptop_output = new hbDNNTensor[visionEncoder_output_count];
    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbDNNGetOutputTensorProperties(&visionEncoder_laptop_output[i].properties, visionEncoder_dnn_handle, i);
        RDK_CHECK_SUCCESS(
            hbSysAllocCachedMem(&visionEncoder_laptop_output[i].sysMem[0], visionEncoder_laptop_output[i].properties.alignedByteSize),
            "BPU ACT Policy VisionEncoder laptop_output hbSysAllocCachedMem failed");
    }

    // phone
    visionEncoder_phone_input = new hbDNNTensor[visionEncoder_input_count];
    for (int i = 0; i < visionEncoder_input_count; i++)
    {
        hbDNNGetInputTensorProperties(&visionEncoder_phone_input[i].properties, visionEncoder_dnn_handle, i);
        RDK_CHECK_SUCCESS(
            hbSysAllocCachedMem(&visionEncoder_phone_input[i].sysMem[0], visionEncoder_phone_input[i].properties.alignedByteSize),
            "BPU ACT Policy VisionEncoder phone_input hbSysAllocCachedMem failed");
    }

    visionEncoder_phone_output = new hbDNNTensor[visionEncoder_output_count];
    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbDNNGetOutputTensorProperties(&visionEncoder_phone_output[i].properties, visionEncoder_dnn_handle, i);
        RDK_CHECK_SUCCESS(
            hbSysAllocCachedMem(&visionEncoder_phone_output[i].sysMem[0], visionEncoder_phone_output[i].properties.alignedByteSize),
            "BPU ACT Policy VisionEncoder phone_output hbSysAllocCachedMem failed");
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
    
    // 模型输出检查
    transformerLayers_output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&transformerLayers_output_count, transformerLayers_dnn_handle),
        "BPU ACT Policy TransformerLayers hbDNNGetOutputCount failed");
    if (transformerLayers_output_count != 1)
        throw std::runtime_error("Your BPU ACT Policy TransformerLayers Model output num is not 1, can't use this python api.");

    transformerLayers_input = new hbDNNTensor[transformerLayers_input_count];
    for (int i = 0; i < transformerLayers_input_count; i++)
    {
        hbDNNGetInputTensorProperties(&transformerLayers_input[i].properties, transformerLayers_dnn_handle, i);
        if (i == 0)
        {
            RDK_CHECK_SUCCESS(
                hbSysAllocCachedMem(&transformerLayers_input[i].sysMem[0], transformerLayers_input[i].properties.alignedByteSize),
                "BPU ACT Policy TransformerLayers laptop_input hbSysAllocCachedMem failed");
        }
        if (i == 1)
        {
            transformerLayers_input[i].sysMem[0] = visionEncoder_laptop_output[0].sysMem[0];
        }
        if (i == 2)
        {
            transformerLayers_input[i].sysMem[0] = visionEncoder_phone_output[0].sysMem[0];
        }
    }

    transformerLayers_output = new hbDNNTensor[transformerLayers_output_count];
    for (int i = 0; i < transformerLayers_output_count; i++)
    {
        hbDNNGetOutputTensorProperties(&transformerLayers_output[i].properties, transformerLayers_dnn_handle, i);
        RDK_CHECK_SUCCESS(
            hbSysAllocCachedMem(&transformerLayers_output[i].sysMem[0], transformerLayers_output[i].properties.alignedByteSize),
            "BPU ACT Policy TransformerLayers laptop_output hbSysAllocCachedMem failed");
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
    float *laptop_hbTensor_ptr = reinterpret_cast<float *>(visionEncoder_laptop_input[0].sysMem[0].virAddr);
    std::memcpy(laptop_hbTensor_ptr, laptop_np_ptr, laptop_buf_info.size * sizeof(float));

    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbSysFlushMem(&visionEncoder_laptop_input[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    hbDNNTaskHandle_t laptop_task_handle = nullptr;
    hbDNNInferCtrlParam laptop_ctrl_param;

    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&laptop_ctrl_param);
    RDK_CHECK_SUCCESS(hbDNNInfer(&laptop_task_handle, &visionEncoder_laptop_output, visionEncoder_laptop_input, visionEncoder_dnn_handle, &laptop_ctrl_param),
                      "BPU ACT Policy Vision Encoder laptop_task hbDNNInfer failed");

    // 启动 phone 的 VisonEncoder 特征推理
    float *phone_np_ptr = reinterpret_cast<float *>(phone_buf_info.ptr);
    float *phone_hbTensor_ptr = reinterpret_cast<float *>(visionEncoder_phone_input[0].sysMem[0].virAddr);
    std::memcpy(phone_hbTensor_ptr, phone_np_ptr, phone_buf_info.size * sizeof(float));

    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbSysFlushMem(&visionEncoder_phone_input[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    hbDNNTaskHandle_t phone_task_handle = nullptr;
    hbDNNInferCtrlParam phone_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&phone_ctrl_param);

    RDK_CHECK_SUCCESS(hbDNNInfer(&phone_task_handle, &visionEncoder_phone_output, visionEncoder_phone_input, visionEncoder_dnn_handle, &phone_ctrl_param),
                      "BPU ACT Policy Vision Encoder phone_task hbDNNInfer failed");

    // 为 TransformerLayers 准备输入数据
    float *state_np_ptr = reinterpret_cast<float *>(state_buf_info.ptr);
    float *state_hbTensor_ptr = reinterpret_cast<float *>(transformerLayers_input[0].sysMem[0].virAddr);
    std::memcpy(state_hbTensor_ptr, state_np_ptr, state_buf_info.size * sizeof(float));

    // 等待 laptop 的推理结束
    RDK_CHECK_SUCCESS(hbDNNWaitTaskDone(laptop_task_handle, 0),
                      "BPU ACT Policy Vision Encoder laptop_task hbDNNWaitTaskDone failed");

    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbSysFlushMem(&visionEncoder_laptop_output[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // laptop_hbTensor_ptr = reinterpret_cast<float *>(visionEncoder_laptop_output[0].sysMem.virAddr);
    RDK_CHECK_SUCCESS(hbDNNReleaseTask(laptop_task_handle), "BPU ACT Policy Vision Encoder laptop_task hbDNNReleaseTask failed");

    // 等待 phone 的推理结束
    RDK_CHECK_SUCCESS(hbDNNWaitTaskDone(phone_task_handle, 0),
                      "BPU ACT Policy Vision Encoder phone_task hbDNNWaitTaskDone failed");

    for (int i = 0; i < visionEncoder_output_count; i++)
    {
        hbSysFlushMem(&visionEncoder_phone_output[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // phone_hbTensor_ptr = reinterpret_cast<float *>(visionEncoder_phone_output[0].sysMem.virAddr);
    RDK_CHECK_SUCCESS(hbDNNReleaseTask(phone_task_handle), "BPU ACT Policy Vision Encoder phone_task hbDNNReleaseTask failed");

    // 启动 TransformerLayers 的推理
    for (int i = 0; i < transformerLayers_input_count; i++)
    {
        hbSysFlushMem(&transformerLayers_input[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    hbDNNTaskHandle_t transformerLayers_task_handle = nullptr;
    hbDNNInferCtrlParam transformerLayers_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&transformerLayers_ctrl_param);
    RDK_CHECK_SUCCESS(hbDNNInfer(&transformerLayers_task_handle, 
                                 &transformerLayers_output, 
                                 transformerLayers_input, 
                                 transformerLayers_dnn_handle, 
                                 &transformerLayers_ctrl_param),
                      "BPU ACT Policy TransformerLayers hbDNNInfer failed");

    // 等待 TransformerLayers 推理结束
    RDK_CHECK_SUCCESS(hbDNNWaitTaskDone(transformerLayers_task_handle, 0),
                      "BPU ACT Policy TransformerLayers hbDNNWaitTaskDone failed");

    for (int i = 0; i < transformerLayers_output_count; i++)
    {
        hbSysFlushMem(&transformerLayers_output[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    auto actions_hbTensor_ptr = reinterpret_cast<float *>(transformerLayers_output[0].sysMem[0].virAddr);
    RDK_CHECK_SUCCESS(hbDNNReleaseTask(transformerLayers_task_handle), "BPU ACT Policy TransformerLayers hbDNNReleaseTask failed");

    // 返回多维 NumPy 数组

    return py::array_t<float>(shape, actions_hbTensor_ptr);
}


// //////////////////////////////////////////////////////////////////////////////////////////////////////

// ACTPloicyRGBEncoder::ACTPloicyRGBEncoder(std::string &model_path)
// {
//     // 加载模型
//     auto modelFileName = model_path.c_str();
//     RDK_CHECK_SUCCESS(
//         hbDNNInitializeFromFiles(&packed_dnn_handle, &modelFileName, 1),
//         "hbDNNInitializeFromFiles failed");

//     model_count = 0;
//     RDK_CHECK_SUCCESS(
//         hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
//         "hbDNNGetModelNameList failed");

//     model_name = model_name_list[0];
//     RDK_CHECK_SUCCESS(
//         hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
//         "hbDNNGetModelHandle failed");

//     // 模型输入检查
//     input_count = 0;
//     RDK_CHECK_SUCCESS(
//         hbDNNGetInputCount(&input_count, dnn_handle),
//         "hbDNNGetInputCount failed");
//     RDK_CHECK_SUCCESS(
//         hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
//         "hbDNNGetInputTensorProperties failed");
//     if (input_count > 1) // 如果有多个输入, 则不是这套接口可推理的模型
//         throw std::runtime_error("Your Model have more than 1 input, can't use this python api.");
//     if (input_properties.tensorType != HB_DNN_TENSOR_TYPE_F32)
//         throw std::runtime_error("Your Model input type is not HB_DNN_TENSOR_TYPE_F32, can't use this python api.");
//     if (input_properties.tensorLayout != HB_DNN_LAYOUT_NCHW)
//         throw std::runtime_error("Your Model input layput is not HB_DNN_LAYOUT_NCHW, can't use this python api.");
//     if (input_properties.validShape.numDimensions != 4)
//         throw std::runtime_error("Your Model input validShape.numDimensions is not 4, can't use this python api.");
//     if (input_properties.validShape.dimensionSize[0] != 1 ||
//         input_properties.validShape.dimensionSize[1] != 3 ||
//         input_properties.validShape.dimensionSize[2] != 480 ||
//         input_properties.validShape.dimensionSize[3] != 640)
//         throw std::runtime_error("Your Model input shape is not (1,3,480,640), can't use this python api.");

//     // 模型输出检查
//     output_count = 0;
//     RDK_CHECK_SUCCESS(
//         hbDNNGetOutputCount(&output_count, dnn_handle),
//         "hbDNNGetOutputCount failed");
//     RDK_CHECK_SUCCESS(
//         hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, 0),
//         "hbDNNGetInputTensorProperties failed");
//     if (input_count != 1)
//         throw std::runtime_error("Your Model have more than 1 output, can't use this python api.");
//     if (output_properties.tensorType != HB_DNN_TENSOR_TYPE_F32)
//         throw std::runtime_error("Your Model input type is not HB_DNN_TENSOR_TYPE_F32, can't use this python api.");
//     if (output_properties.tensorLayout != HB_DNN_LAYOUT_NCHW)
//         throw std::runtime_error("Your Model input layput is not HB_DNN_LAYOUT_NCHW, can't use this python api.");
//     if (output_properties.validShape.numDimensions != 4)
//         throw std::runtime_error("Your Model input validShape.numDimensions is not 4, can't use this python api.");
//     if (output_properties.validShape.dimensionSize[0] != 1 ||
//         output_properties.validShape.dimensionSize[1] != 512 ||
//         output_properties.validShape.dimensionSize[2] != 15 ||
//         output_properties.validShape.dimensionSize[3] != 20)
//         throw std::runtime_error("Your Model input shape is not (1,3,480,640), can't use this python api.");

//     // 输入数据的tensor

//     input.properties = input_properties;
//     hbSysAllocCachedMem(&input.sysMem[0], input_properties.alignedByteSize);

//     // 输出数据的tensor
//     output = new hbDNNTensor[output_count];
//     for (int i = 0; i < output_count; i++)
//     {
//         hbDNNTensorProperties &output_properties = output[i].properties;
//         hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
//         int out_aligned_size = output_properties.alignedByteSize;
//         hbSysMem &mem = output[i].sysMem[0];
//         hbSysAllocCachedMem(&mem, out_aligned_size);
//     }
// }

// ACTPloicyRGBEncoder::~ACTPloicyRGBEncoder()
// {
//     hbSysFreeMem(&(input.sysMem[0]));
//     for (int i = 0; i < output_count; i++)
//         hbSysFreeMem(&(output[i].sysMem[0]));
//     hbDNNRelease(packed_dnn_handle);
//     std::cout << "[INFO] release model success." << std::endl;
// }

// py::array_t<float> ACTPloicyRGBEncoder::inference(py::array_t<float> input_array)
// {
//     // np 2 dnnTensor
//     py::buffer_info buf_info = input_array.request(); // 获取输入 NumPy 数组的缓冲区信息
//     if (buf_info.ndim != 4 ||                         // 验证输入数组的形状是否为 (1, 3, 480, 640)
//         buf_info.shape[0] != 1 ||
//         buf_info.shape[1] != 3 ||
//         buf_info.shape[2] != 480 ||
//         buf_info.shape[3] != 640)
//         throw std::runtime_error("Input array must have shape (1, 3, 480, 640).");

//     float *np_ptr = reinterpret_cast<float *>(buf_info.ptr);
//     float *hbTensor_ptr = reinterpret_cast<float *>(input.sysMem[0].virAddr);
//     std::memcpy(hbTensor_ptr, np_ptr, buf_info.size * sizeof(float));

//     hbDNNTaskHandle_t task_handle = nullptr;
//     hbDNNInferCtrlParam infer_ctrl_param;
//     HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
//     hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);
//     hbDNNWaitTaskDone(task_handle, 0);

//     hbSysFlushMem(&(output[0].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
//     hbTensor_ptr = reinterpret_cast<float *>(output[0].sysMem[0].virAddr);

//     hbDNNReleaseTask(task_handle);

//     std::vector<size_t> shape = {1, 512, 15, 20};

//     // 返回多维 NumPy 数组
//     return py::array_t<float>(
//         shape,       // 输出数组的形状
//         hbTensor_ptr // 输出数组的数据指针
//     );

// }
