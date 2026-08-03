/*
 * Copyright (C) 2024 Shanghai Gua Technology Co., Ltd.
 * All rights reserved
 */
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "utils/common.h"
#include "utils/model_manager.h"

#ifndef XLM_ENABLE_TIME_PROFILING
#define XLM_ENABLE_TIME_PROFILING 0
#endif

/**
 * @brief 时间性能分析宏
 *
 * @param name 时间性能分析名称
 * @param msg 时间性能分析消息
 */
#if XLM_ENABLE_TIME_PROFILING
#define TIME_PROFILE_BEGIN(name) \
  auto name##_begin_time = xlm::utils::GetSteadyTimeMicros();
#define TIME_PROFILE_END(name, msg)                           \
  do {                                                        \
    auto name##_end_time = xlm::utils::GetSteadyTimeMicros(); \
    LOGI("TimeProfile", msg " cost time: {} us",              \
         name##_end_time - name##_begin_time);                \
  } while (0);
#else
#define TIME_PROFILE_BEGIN(name)
#define TIME_PROFILE_END(name, msg)
#endif

namespace xlm::utils {

template <typename Enum>
constexpr auto AsNum(Enum const e) noexcept -> std::underlying_type<Enum> {
  return static_cast<std::underlying_type<Enum>>(e);
}

template <typename T>
static inline constexpr bool IsZero(T const value) {
  static_assert(std::is_floating_point_v<T>,
                "Only float and double are supported");
  return (std::abs(value) < std::numeric_limits<T>::epsilon());
}

/**
 * @brief 获取当前系统时间，毫秒
 *
 * @return 返回当前系统时间，毫秒
 */
static inline int64_t GetSystemTimeMillis() noexcept {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

/**
 * @brief 获取当前系统时间，微秒
 *
 * @return 返回当前系统时间，微秒
 */
static inline int64_t GetSystemTimeMicros() noexcept {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

/**
 * @brief 获取当前稳定时间，毫秒
 *
 * @return 返回当前稳定时间，毫秒
 */
static inline int64_t GetSteadyTimeMillis() noexcept {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/**
 * @brief 获取当前稳定时间，微秒
 *
 * @return 返回当前稳定时间，微秒
 */
static inline int64_t GetSteadyTimeMicros() noexcept {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int32_t DumpData(std::string const &file_prefix, void const *const data,
                 size_t const len);

int32_t ArgSortToEigenVector(std::vector<int32_t> const &data,
                             Eigen::VectorXi &index, bool const is_asc = true);

std::vector<InferBackend> ConvertIntToInferBackend(
    std::vector<int32_t> const &bpu_core);

bool IsCompleteUtf8Sequence(std::string const &sequence);

}  // namespace xlm::utils
