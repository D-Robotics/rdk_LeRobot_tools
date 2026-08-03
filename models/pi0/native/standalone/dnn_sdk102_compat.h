#pragma once

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"

struct hbDNNInferV3Param {
  bool enable_pre_submit = false;
  bool enable_poll = false;
};

static inline int32_t hbDNNInferV3(hbUCPTaskHandle_t* task_handle,
                                   hbDNNTensor* output,
                                   const hbDNNTensor* input,
                                   hbDNNHandle_t dnn_handle,
                                   const hbDNNInferV3Param*) {
  return hbDNNInferV2(task_handle, output, input, dnn_handle);
}

extern "C" int32_t hbDNNNotifyTask(hbUCPTaskHandle_t task_handle);
