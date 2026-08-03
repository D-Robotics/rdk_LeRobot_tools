/*
 * Copyright (C) 2024 Shanghai Gua Technology Co., Ltd.
 * All rights reserved
 */

#include "utils/xlm_utils.h"

#include <algorithm>
#include <codecvt>
#include <cwchar>
#include <iostream>
#include <locale>
#include <string>
#include <unordered_map>
#include <vector>

#define TAG "XLM_UTILS"

static const std::unordered_map<int32_t, xlm::utils::InferBackend>
    kIntToInferBackendMap = {
        {0, xlm::utils::InferBackend::kInferBackendBPU0},
        {1, xlm::utils::InferBackend::kInferBackendBPU1},
        {2, xlm::utils::InferBackend::kInferBackendBPU2},
        {3, xlm::utils::InferBackend::kInferBackendBPU3},
};

namespace xlm::utils {

int32_t DumpData(std::string const &file_prefix, void const *const data,
                 size_t const len) {
  std::string const &file_name{file_prefix + ".bin"};
  std::ofstream fout(file_name.c_str(), std::ios::binary | std::ios::trunc);
  if (!fout.good()) {
    fout.close();
    std::cerr << "Open dump file '" << file_name << "' failed!" << std::endl;
    return -1;
  }
  fout.write(reinterpret_cast<const char *>(data), len);
  fout.close();
  std::cout << "Dump data to file '" << file_name << "' success!" << std::endl;
  return 0;
}

int32_t ArgSortToEigenVector(std::vector<int32_t> const &data,
                             Eigen::VectorXi &index, bool const is_asc) {
  index.resize(data.size());
  for (int32_t i = 0; i < static_cast<int32_t>(data.size()); ++i) {
    index(i) = i;
  }

  std::sort(index.data(), index.data() + index.size(),
            [&data, &is_asc](int32_t const a, int32_t const b) {
              return is_asc ? data[a] < data[b] : data[a] > data[b];
            });

  return 0;
}

std::vector<InferBackend> ConvertIntToInferBackend(
    std::vector<int32_t> const &bpu_core) {
  std::vector<InferBackend> infer_backend(bpu_core.size());

  if (0 == bpu_core.size()) {
    LOGW(TAG, "bpu_core is empty, use default bpu 0 core");
    infer_backend.emplace_back(InferBackend::kInferBackendBPU0);
    return infer_backend;
  }

  for (int32_t i = 0; i < static_cast<int32_t>(bpu_core.size()); ++i) {
    auto const &iter = kIntToInferBackendMap.find(bpu_core[i]);
    if (iter == kIntToInferBackendMap.end()) {
      LOGW(TAG, "bpu_core[{}] is invalid, use default bpu 0 core", i);
      infer_backend[i] = InferBackend::kInferBackendBPU0;
    } else {
      infer_backend[i] = iter->second;
    }
  }

  return infer_backend;
}

bool IsCompleteUtf8Sequence(std::string const &sequence) {
  if (sequence.empty()) {
    return true;
  }

  // 解码整个序列并检查是否会产生替换字符或无效字符
  try {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::wstring ws =
        conv.from_bytes(sequence);  // 解码整个序列，不只是最后一个字符

    // 检查每个解码后的字符
    for (wchar_t ch : ws) {
      // 替换字符 (U+FFFD)
      if (ch == 0xFFFD) {
        return false;
      }
      // Surrogate 区 (0xD800-0xDFFF) - 不应该出现在 UTF-8 中
      if (ch >= 0xD800 && ch <= 0xDFFF) {
        return false;
      }
      // 超出 Unicode 范围
      if (ch > 0x10FFFF) {
        return false;
      }
    }
  } catch (const std::range_error &) {
    // 解码失败，说明序列无效
    return false;
  } catch (...) {
    // 其他异常，保守处理，返回 false
    return false;
  }

  return true;
}

}  // namespace xlm::utils
