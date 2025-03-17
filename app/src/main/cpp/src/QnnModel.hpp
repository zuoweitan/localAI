#ifndef QNNMODEL_HPP
#define QNNMODEL_HPP

#include <QnnSampleApp.hpp>
#include <QnnTypeMacros.hpp>
#include <Config.hpp>

#include <inttypes.h>

#include <cstring>
#include <fstream>
#include <iostream>

#include "DataUtil.hpp"
#include "Logger.hpp"
#include "SDUtils.hpp"

using namespace qnn::tools::sample_app;

class QnnModel : public QnnSampleApp
{
public:
  QnnModel(QnnFunctionPointers &qnnFunctionPointers,
           std::string &backendPath,
           std::string &modelPath,
           void *&contextHandle,
           std::string &backendExtensionPath,
           bool &loadFromCachedBinary,
           qnn::tools::iotensor::OutputDataType &outputDataType,
           qnn::tools::iotensor::InputDataType &inputDataType,
           qnn::tools::sample_app::ProfilingLevel &profilingLevel,
           bool debugMode,
           std::string &logPath,
           std::string &targetArchitecture)
      : QnnSampleApp(qnnFunctionPointers,
                     backendPath,
                     modelPath,
                     contextHandle,
                     backendExtensionPath,
                     loadFromCachedBinary,
                     outputDataType,
                     inputDataType,
                     profilingLevel,
                     debugMode,
                     logPath,
                     targetArchitecture) {}

  StatusCode executeClipOnce(
      ClipInput &clip_input,
      UnetInput &clip_output,
      Qnn_Tensor_t *inputs,
      Qnn_Tensor_t *outputs,
      const qnn_wrapper_api::GraphInfo &graphInfo,
      size_t input_offset,
      size_t output_offset)
  {
    // input_ids
    uint32_t elementCount = QNN_TENSOR_GET_DIMENSIONS(inputs[0])[0] *
                            QNN_TENSOR_GET_DIMENSIONS(inputs[0])[1];
    memcpy(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data,
           clip_input.input_ids.data() + input_offset,
           elementCount * sizeof(int32_t));

    // execute graph
    QNN_DEBUG("Executing Clip");
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph,
        inputs,
        graphInfo.numInputTensors,
        outputs,
        graphInfo.numOutputTensors,
        m_profileBackendHandle,
        nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    QNN_INFO("Graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus)
    {
      QNN_ERROR("Graph execution failed!");
      return StatusCode::FAILURE;
    }

    // get output
    elementCount = QNN_TENSOR_GET_DIMENSIONS(outputs[0])[0] *
                   QNN_TENSOR_GET_DIMENSIONS(outputs[0])[1] *
                   QNN_TENSOR_GET_DIMENSIONS(outputs[0])[2];

    // memcpy(clip_output.text_embedding.data() + output_offset,
    //        QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data,
    //        elementCount * sizeof(uint16_t));

    float *tmp = nullptr;
    if (qnn::tools::iotensor::StatusCode::SUCCESS != m_ioTensor.convertToFloat(&tmp, &outputs[0]))
    {
      return StatusCode::FAILURE;
    }

    memcpy(clip_output.text_embedding_float.data() + output_offset,
           tmp,
           elementCount * sizeof(float));
    free(tmp);

    return StatusCode::SUCCESS;
  }

  StatusCode executeUnetOnce(
      UnetInput &unet_input,
      UnetOutput &unet_output,
      Qnn_Tensor_t *inputs,
      Qnn_Tensor_t *outputs,
      const qnn_wrapper_api::GraphInfo &graphInfo,
      size_t latent_offset,
      size_t text_embedding_offset)
  {
    // latents
    uint16_t *latents = static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
    int elementCount = 1 * 4 * sample_size * sample_size;
    qnn::tools::datautil::floatToTfN(latents,
                                     unet_input.latents.data() + latent_offset,
                                     inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
                                     inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale,
                                     elementCount);

    // position
    int32_t *positionData = static_cast<int32_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[1]).data);
    positionData[0] = unet_input.timestep;

    // text_embedding
    // elementCount = 1 * 77 * text_embedding_size;
    // memcpy(QNN_TENSOR_GET_CLIENT_BUF(inputs[2]).data,
    //        unet_input.text_embedding.data() + text_embedding_offset,
    //        elementCount * sizeof(uint16_t));
    elementCount = 1 * 77 * text_embedding_size;
    uint16_t *text_embedding_float = static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[2]).data);
    qnn::tools::datautil::floatToTfN(text_embedding_float,
                                     unet_input.text_embedding_float.data() + text_embedding_offset,
                                     inputs[2].v1.quantizeParams.scaleOffsetEncoding.offset,
                                     inputs[2].v1.quantizeParams.scaleOffsetEncoding.scale,
                                     elementCount);
    // execute graph
    QNN_DEBUG("Executing Unet");
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph,
        inputs,
        graphInfo.numInputTensors,
        outputs,
        graphInfo.numOutputTensors,
        m_profileBackendHandle,
        nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    QNN_INFO("Decoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus)
    {
      QNN_ERROR("Decoder graph execution failed!");
      return StatusCode::FAILURE;
    }

    // get output
    float *tmp = nullptr;
    if (qnn::tools::iotensor::StatusCode::SUCCESS != m_ioTensor.convertToFloat(&tmp, &outputs[0]))
    {
      return StatusCode::FAILURE;
    }

    // const size_t single_result_size = unet_output.latents.size() / (output_offset > 0 ? 2 : 1);
    elementCount = 1 * 4 * sample_size * sample_size;
    memcpy(unet_output.latents.data() + latent_offset,
           tmp,
           elementCount * sizeof(float));
    free(tmp);

    return StatusCode::SUCCESS;
  }

  StatusCode executeClipGraphs(
      ClipInput &clip_input,
      UnetInput &clip_output,
      bool use_cfg)
  {
    auto returnStatus = StatusCode::SUCCESS;

    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++)
    {
      QNN_DEBUG("Starting execution for graphIdx: %d", graphIdx);

      Qnn_Tensor_t *inputs = nullptr;
      Qnn_Tensor_t *outputs = nullptr;

      // set input/output tensor
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs, (*m_graphsInfo)[graphIdx]))
      {
        QNN_ERROR("Error in setting up Input and output Tensors for graphIdx: %d", graphIdx);
        returnStatus = StatusCode::FAILURE;
        break;
      }

      auto graphInfo = (*m_graphsInfo)[graphIdx];

      // check
      if (graphInfo.numInputTensors != 1 || graphInfo.numOutputTensors != 1)
      {
        QNN_ERROR("Expecting 1 input and 1 output tensor, got %d inputs and %d outputs",
                  graphInfo.numInputTensors, graphInfo.numOutputTensors);
        returnStatus = StatusCode::FAILURE;
        break;
      }

      if (!use_cfg)
      {
        // execute once
        returnStatus = executeClipOnce(clip_input, clip_output, inputs, outputs, graphInfo, 0, 0);
      }
      else
      {
        // execute twice
        const uint32_t elementCount = QNN_TENSOR_GET_DIMENSIONS(inputs[0])[0] *
                                      QNN_TENSOR_GET_DIMENSIONS(inputs[0])[1];

        // first execution
        returnStatus = executeClipOnce(clip_input, clip_output, inputs, outputs, graphInfo, 0, 0);
        if (returnStatus != StatusCode::SUCCESS)
        {
          break;
        }
        // second execution
        returnStatus = executeClipOnce(clip_input, clip_output, inputs, outputs, graphInfo, elementCount, 77 * text_embedding_size);
      }

      // clean up
      m_ioTensor.tearDownInputAndOutputTensors(inputs,
                                               outputs,
                                               graphInfo.numInputTensors,
                                               graphInfo.numOutputTensors);
      inputs = nullptr;
      outputs = nullptr;

      if (StatusCode::SUCCESS != returnStatus)
      {
        break;
      }
    }

    return returnStatus;
  }
  StatusCode executeUnetGraphsFirst(
      UnetInput &unet_input,
      UnetOutput &unet_output,
      Qnn_Tensor_t *&inputs_ptr,
      Qnn_Tensor_t *&outputs_ptr,
      bool use_cfg)
  {
    auto returnStatus = StatusCode::SUCCESS;
    Qnn_Tensor_t *inputs = nullptr;
    Qnn_Tensor_t *outputs = nullptr;

    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++)
    {
      QNN_DEBUG("Starting decoder execution for graphIdx: %d", graphIdx);

      // set input/output tensor
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs, (*m_graphsInfo)[graphIdx]))
      {
        QNN_ERROR("Error in setting up Input and output Tensors for graphIdx: %d", graphIdx);
        returnStatus = StatusCode::FAILURE;
        break;
      }
      inputs_ptr = inputs;
      outputs_ptr = outputs;

      auto graphInfo = (*m_graphsInfo)[graphIdx];

      if (graphInfo.numInputTensors != 3)
      {
        QNN_ERROR("Expecting 3 input tensors, got %d", graphInfo.numInputTensors);
        returnStatus = StatusCode::FAILURE;
        break;
      }

      // const size_t single_result_size = unet_output.latents.size() / (use_cfg ? 2 : 1);

      if (!use_cfg)
      {
        // execute once
        returnStatus = executeUnetOnce(unet_input, unet_output, inputs, outputs, graphInfo, 0, 0);
      }
      else
      {
        // execute twice
        // first execution
        returnStatus = executeUnetOnce(unet_input, unet_output, inputs, outputs, graphInfo, 0, 0);
        if (returnStatus != StatusCode::SUCCESS)
        {
          break;
        }

        // second execution
        returnStatus = executeUnetOnce(unet_input, unet_output, inputs, outputs, graphInfo, 1 * 4 * sample_size * sample_size, 1 * 77 * text_embedding_size);
      }

      if (StatusCode::SUCCESS != returnStatus)
      {
        break;
      }
    }

    return returnStatus;
  }
  StatusCode executeUnetGraphsRemain(
      UnetInput &unet_input,
      UnetOutput &unet_output,
      Qnn_Tensor_t *&inputs,
      Qnn_Tensor_t *&outputs,
      bool use_cfg)
  {
    auto returnStatus = StatusCode::SUCCESS;

    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++)
    {
      QNN_DEBUG("Starting decoder execution for graphIdx: %d", graphIdx);

      auto graphInfo = (*m_graphsInfo)[graphIdx];

      if (graphInfo.numInputTensors != 3)
      {
        QNN_ERROR("Expecting 3 input tensors, got %d", graphInfo.numInputTensors);
        returnStatus = StatusCode::FAILURE;
        break;
      }

      // const size_t single_result_size = unet_output.latents.size() / (use_cfg ? 2 : 1);

      if (!use_cfg)
      {
        // execute once
        returnStatus = executeUnetOnce(unet_input, unet_output, inputs, outputs, graphInfo, 0, 0);
      }
      else
      {
        // execute twice
        // first execution
        returnStatus = executeUnetOnce(unet_input, unet_output, inputs, outputs, graphInfo, 0, 0);
        if (returnStatus != StatusCode::SUCCESS)
        {
          break;
        }

        // second execution
        returnStatus = executeUnetOnce(unet_input, unet_output, inputs, outputs, graphInfo, 1 * 4 * sample_size * sample_size, 1 * 77 * text_embedding_size);
      }

      if (StatusCode::SUCCESS != returnStatus)
      {
        break;
      }
    }

    return returnStatus;
  }
  StatusCode executeVaeDecoderGraphs(VaeDecoderInput &decoder_input, Picture &decoder_output)
  {
    auto returnStatus = StatusCode::SUCCESS;

    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++)
    {
      QNN_DEBUG("Starting decoder execution for graphIdx: %d", graphIdx);

      Qnn_Tensor_t *inputs = nullptr;
      Qnn_Tensor_t *outputs = nullptr;
      // set input/output tensor
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs, (*m_graphsInfo)[graphIdx]))
      {
        QNN_ERROR("Error in setting up Input and output Tensors for graphIdx: %d", graphIdx);
        returnStatus = StatusCode::FAILURE;
        break;
      }

      auto graphInfo = (*m_graphsInfo)[graphIdx];

      if (graphInfo.numInputTensors != 1)
      {
        QNN_ERROR("Expecting 1 input tensors, got %d", graphInfo.numInputTensors);
        returnStatus = StatusCode::FAILURE;
        break;
      }

      // latents
      {
        uint16_t *latents = static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
        qnn::tools::datautil::floatToTfN(latents,
                                         decoder_input.latents.data(),
                                         inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
                                         inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale,
                                         decoder_input.latents.size());
        // memcpy(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data,
        //        decoder_input.latents.data(),
        //        decoder_input.latents.size() * sizeof(uint16_t));
      }

      // execute graph
      QNN_DEBUG("Executing decoder graph: %d", graphIdx);
      auto start_time = std::chrono::high_resolution_clock::now();

      auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
          graphInfo.graph,
          inputs,
          graphInfo.numInputTensors,
          outputs,
          graphInfo.numOutputTensors,
          m_profileBackendHandle,
          nullptr);

      auto end_time = std::chrono::high_resolution_clock::now();
      int duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      QNN_INFO("Decoder graph execution time: %d ms", duration);

      if (QNN_GRAPH_NO_ERROR != executeStatus)
      {
        returnStatus = StatusCode::FAILURE;
        QNN_ERROR("Decoder graph execution failed!");
      }

      // get output
      if (StatusCode::SUCCESS == returnStatus)
      {
        {
          float *tmp = nullptr;
          float *latentsData = decoder_output.pixel_values.data();
          if (qnn::tools::iotensor::StatusCode::SUCCESS != m_ioTensor.convertToFloat(&tmp, &outputs[0]))
          {
            returnStatus = StatusCode::FAILURE;
            break;
          }
          memcpy(latentsData, tmp, decoder_output.pixel_values.size() * sizeof(float));
          free(tmp);
        }
      }

      if (StatusCode::SUCCESS != returnStatus)
      {
        break;
      }
    }

    //  qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    //  m_graphsInfo = nullptr;
    return returnStatus;
  }
  StatusCode cleanUnetGraphs(Qnn_Tensor_t *inputs, Qnn_Tensor_t *outputs)
  {
    auto returnStatus = StatusCode::SUCCESS;

    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++)
    {
      QNN_DEBUG("Starting decoder cleaning for graphIdx: %d", graphIdx);

      auto graphInfo = (*m_graphsInfo)[graphIdx];
      m_ioTensor.tearDownInputAndOutputTensors(inputs,
                                               outputs,
                                               graphInfo.numInputTensors,
                                               graphInfo.numOutputTensors);
      inputs = nullptr;
      outputs = nullptr;

      if (StatusCode::SUCCESS != returnStatus)
      {
        break;
      }
    }
    //  qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    //  m_graphsInfo = nullptr;
    return returnStatus;
  }
};

#endif // QNNMODEL_HPP