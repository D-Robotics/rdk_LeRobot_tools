

from libpycauchyX5tools import BPU_ACTPolicy
import onnxruntime
import numpy as np
import os
import cv2
import os
import shutil
from time import time
import cv2
import numpy as np
import argparse
import logging 

from time import time

def ave_error(tensor1, tensor2):
    """
    比较两个 NumPy 数组的误差。
    参数:
        tensor1: 第一个 NumPy 数组。
        tensor2: 第二个 NumPy 数组。
    返回:
        一个字典，包含多种误差指标。
    """
    # 确保输入是 NumPy 数组
    tensor1 = np.array(tensor1.copy())
    tensor2 = np.array(tensor2.copy())
    # 检查形状是否一致
    if tensor1.shape != tensor2.shape:
        raise ValueError("两个数组的形状必须相同！")
    # 计算误差
    diff = tensor1 - tensor2
    # 均方误差 (MSE)
    mse = np.mean(diff**2)
    # 余弦相似度
    cosine_similarity = np.dot(tensor1.flatten(), tensor2.flatten()) / (
        np.linalg.norm(tensor1) * np.linalg.norm(tensor2) + 1e-8
    )
    # 返回结果
    return mse, cosine_similarity
def main():

    float_onnx_visionEncoder = onnxruntime.InferenceSession("cauchy_test4/BPU_ACTPolicy_VisionEncoder/BPU_ACTPolicy_VisionEncoder.onnx")
    float_onnx_transformerLayers = onnxruntime.InferenceSession("cauchy_test4/BPU_ACTPolicy_TransformerLayers/BPU_ACTPolicy_TransformerLayers.onnx")
    
    policy = BPU_ACTPolicy("cauchy_test4/BPU_ACTPolicy_VisionEncoder/bpu_model_output/BPU_ACTPolicy_VisionEncoder.bin", "cauchy_test4/BPU_ACTPolicy_TransformerLayers/bpu_model_output/BPU_ACTPolicy_TransformerLayers.bin")
    
    for k in range(400):
        if k % 4 != 0:
            continue
        state = np.fromfile("cauchy_test4/BPU_ACTPolicy_TransformerLayers/calibration_data_BPU_ACTPolicy_TransformerLayers/state/%.10d.nchw"%k, dtype=np.float32).reshape(1, 12)
        phone = np.fromfile("cauchy_test4/BPU_ACTPolicy_VisionEncoder/calibration_data_BPU_ACTPolicy_VisionEncoder/phone_%.10d.nchw"%k, dtype=np.float32).reshape(1, 3, 480, 640)
        laptop = np.fromfile("cauchy_test4/BPU_ACTPolicy_VisionEncoder/calibration_data_BPU_ACTPolicy_VisionEncoder/laptop_%.10d.nchw"%k, dtype=np.float32).reshape(1, 3, 480, 640)

        begin_time = time()
        bin_out = policy(state[:,:,np.newaxis, np.newaxis].copy(), phone.copy(), laptop.copy())[:,:,:,0]
        print("## BPU inference time: %.2f ms"%(1000*(time() - begin_time)), end=": ")
        
        
        laptop_feature = float_onnx_visionEncoder.run([float_onnx_visionEncoder.get_outputs()[0].name], {float_onnx_visionEncoder.get_inputs()[0].name: laptop.copy()})[0]
        phone_feature = float_onnx_visionEncoder.run([float_onnx_visionEncoder.get_outputs()[0].name], {float_onnx_visionEncoder.get_inputs()[0].name: phone.copy()})[0]
        actions = float_onnx_transformerLayers.run([float_onnx_transformerLayers.get_outputs()[0].name], {float_onnx_transformerLayers.get_inputs()[0].name: state.copy(),
                                                                                                    float_onnx_transformerLayers.get_inputs()[1].name: laptop_feature.copy(),
                                                                                                    float_onnx_transformerLayers.get_inputs()[2].name: phone_feature.copy()
                                                                                                    })[0]
        # print(f"{actions.shape =}")
        mse, cosine_similarity = ave_error(bin_out, actions)
        print("mse: %.5f, cos_sim: %.5f"%(mse, cosine_similarity))
    
    # state = np.load("bpu_act_so100_0409/BPU_ACTPolicy_TransformerLayers/calibration_data_BPU_ACTPolicy_TransformerLayers/state/0000000008.npy")
    # phone = np.load("bpu_act_so100_0409/BPU_ACTPolicy_VisionEncoder/calibration_data_BPU_ACTPolicy_VisionEncoder/phone_0000000008.npy")
    # laptop = np.load("bpu_act_so100_0409/BPU_ACTPolicy_VisionEncoder/calibration_data_BPU_ACTPolicy_VisionEncoder/laptop_0000000008.npy")

    # begin_time = time()
    # bin_out = policy(state.copy(), phone.copy(), laptop.copy())
    # print("## BPU inference time: %.2f ms"%(1000*(time() - begin_time)))
    
    
    # laptop_feature = float_onnx_visionEncoder.run([float_onnx_visionEncoder.get_outputs()[0].name], {float_onnx_visionEncoder.get_inputs()[0].name: laptop.copy()})[0]
    # phone_feature = float_onnx_visionEncoder.run([float_onnx_visionEncoder.get_outputs()[0].name], {float_onnx_visionEncoder.get_inputs()[0].name: phone.copy()})[0]
    # actions = float_onnx_transformerLayers.run([float_onnx_transformerLayers.get_outputs()[0].name], {float_onnx_transformerLayers.get_inputs()[0].name: state.copy(),
    #                                                                                               float_onnx_transformerLayers.get_inputs()[1].name: laptop_feature.copy(),
    #                                                                                               float_onnx_transformerLayers.get_inputs()[2].name: phone_feature.copy()
    #                                                                                               })[0]
    # print(f"{actions.shape =}")
    # mse, cosine_similarity = ave_error(bin_out, actions)
    # print("mse: %.5f, cos_sim: %.5f"%(mse, cosine_similarity))
    # vision_path = "bpu_act_so100_0409/BPU_ACTPolicy_VisionEncoder/calibration_data_BPU_ACTPolicy_VisionEncoder/"
    # for i, name in enumerate(os.listdir(vision_path)):
    #     phone = np.load(os.path.join(vision_path, name))
    #     begin_time = time()
    #     # bin_out = policy(state, phone, laptop)
    #     bin_out = policy(state.copy(), phone.copy(), laptop.copy())
    #     print("## %d BPU inference time: %.2f ms"%(i, 1000*(time() - begin_time)))
    #     begin_time = time()
    #     onnx_out = float_onnx_visionEncoder.run([float_onnx_visionEncoder.get_outputs()[0].name], {float_onnx_visionEncoder.get_inputs()[0].name: phone.copy()})[0]
    #     print("## %d ONNX inference time: %.2f ms"%(i, 1000*(time() - begin_time)))

    #     mse, cosine_similarity = ave_error(bin_out, onnx_out)
    #     print("mse: %.5f, cos_sim: %.5f"%(mse, cosine_similarity))
if __name__ == "__main__":
    main()