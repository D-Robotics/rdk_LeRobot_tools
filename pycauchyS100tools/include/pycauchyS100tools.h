

#ifndef _PY_CAUCHY_S100_TOOLS_
#define _PY_CAUCHY_S100_TOOLS_

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <utility>
#include <vector>
#include <cstring>
#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "gflags/gflags.h"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"
#define EMPTY ""

// #define HB_CHECK_SUCCESS(value, errmsg) \
//   do                                    \
//   {                                     \
//     /*value can be call of function*/   \
//     auto ret_code = value;              \
//     if (ret_code != 0)                  \
//     {                                   \
//       throw std::runtime_error(errmsg); \
//     }                                   \
//   } while (0);

#define RDK_CHECK_SUCCESS(value, errmsg) \
  do                                     \
  {                                      \
    /*value can be call of function*/    \
    auto ret_code = value;               \
    if (ret_code != 0)                   \
    {                                    \
      throw std::runtime_error(errmsg);  \
    }                                    \
  } while (0);

namespace py = pybind11;

class __attribute__((visibility("default"))) BPU_ACTPolicy
{
public:
  // ACTPloicyRGBEncoder();
  BPU_ACTPolicy(std::string &visionEncoder_model_path, std::string &transformerLayers_model_path);
  ~BPU_ACTPolicy();
  py::array_t<float> inference(py::array_t<float> state, py::array_t<float> laptop, py::array_t<float> phone);

private:
hbDNNPackedHandle_t visionEncoder_packed_dnn_handle;
  hbDNNHandle_t visionEncoder_dnn_handle;
  hbDNNTensorProperties visionEncoder_input_properties;
  hbDNNTensorProperties visionEncoder_output_properties;
  std::vector<hbDNNTensor> visionEncoder_phone_input;
  std::vector<hbDNNTensor> visionEncoder_phone_output;
  std::vector<hbDNNTensor> visionEncoder_laptop_input;
  std::vector<hbDNNTensor> visionEncoder_laptop_output;
  const char **visionEncoder_model_name_list;
  int visionEncoder_model_count;
  const char *visionEncoder_model_name;
  int32_t visionEncoder_input_count;
  int32_t visionEncoder_output_count;

  hbDNNPackedHandle_t transformerLayers_packed_dnn_handle;
  hbDNNHandle_t transformerLayers_dnn_handle;
  hbDNNTensorProperties transformerLayers_input_properties;
  hbDNNTensorProperties transformerLayers_output_properties;
  std::vector<hbDNNTensor> transformerLayers_input;
  std::vector<hbDNNTensor> transformerLayers_output;
  const char **transformerLayers_model_name_list;
  int transformerLayers_model_count;
  const char *transformerLayers_model_name;
  int32_t transformerLayers_input_count;
  int32_t transformerLayers_output_count;

  std::vector<size_t> shape;

  py::array_t<float> result;
};


#endif // _PY_CAUCHY_S100_TOOLS_