/*
 * Copyright (C) 2024 Shanghai Gua Technology Co., Ltd.
 * All rights reserved
 */

#pragma once

#include <vector>

#include "hlog/logging.h"
#include "utils/xlm_utils.h"

#define SET_LOG_LEVEL_TRACE() \
  HLOG_SET_LOGLEVEL(hobot::hlog::LogLevel::log_trace)
#define SET_LOG_LEVEL_DEBUG() \
  HLOG_SET_LOGLEVEL(hobot::hlog::LogLevel::log_debug)
#define SET_LOG_LEVEL_INFO() HLOG_SET_LOGLEVEL(hobot::hlog::LogLevel::log_info)
#define SET_LOG_LEVEL_WARN() HLOG_SET_LOGLEVEL(hobot::hlog::LogLevel::log_warn)
#define SET_LOG_LEVEL_ERROR() HLOG_SET_LOGLEVEL(hobot::hlog::LogLevel::log_err)

#define LOGT(tag, err_msg, ...) HFLOGM_T(tag, err_msg, ##__VA_ARGS__)
#define LOGD(tag, err_msg, ...) HFLOGM_D(tag, err_msg, ##__VA_ARGS__)
#define LOGI(tag, err_msg, ...) HFLOGM_I(tag, err_msg, ##__VA_ARGS__)
#define LOGE(tag, err_msg, ...) HFLOGM_E(tag, err_msg, ##__VA_ARGS__)
#define LOGW(tag, err_msg, ...) HFLOGM_W(tag, err_msg, ##__VA_ARGS__)

enum XlmErrorCode : int32_t { kSuccess = 0, kFailed };

// 展示被调用方的错误码，固定返回 xlm 定义的错误码
#define HB_CHECK_SUCCESS(value, errmsg)               \
  do {                                                \
    /*value can be call of function*/                 \
    auto ret_code{value};                             \
    if (ret_code != XlmErrorCode::kSuccess) {         \
      LOGE("ret_check", "{}, error code: {}", errmsg, \
           static_cast<int32_t>(ret_code));           \
      return kFailed;                                 \
    }                                                 \
  } while (0);

// 自定义错误码
#define LOGE_AND_RETURN_IF(condition, err_msg, code) \
  do {                                               \
    if (condition) {                                 \
      LOGE(TAG, "{}", err_msg);                      \
      return code;                                   \
    }                                                \
  } while (0)
