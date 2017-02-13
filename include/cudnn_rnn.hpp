//
// Created by Aman LaChapelle on 1/10/17.
//
// homeAutomation
// Copyright (c) 2017 Aman LaChapelle
// Full license at homeAutomation/LICENSE.txt
//

/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef HOMEAUTOMATION_CUDNN_RNN_H_HPP
#define HOMEAUTOMATION_CUDNN_RNN_H_HPP

//dlib dependencies
#include <dlib/dnn/tensor.h>
#include <dlib/dnn/cuda_errors.h>
#include <dlib/dnn/cuda_dlib.h>
#include <dlib/dnn/cuda_utils.h>
#include <dlib/dnn/cudnn_dlibapi.h>
#include <dlib/dnn/layers.h>

//cudnn dependencies
#include <cudnn.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const char *cudnn_get_error_string(cudnnStatus_t s) {
  switch (s) {
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDA Runtime API initialization failed.";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDA Resources could not be allocated.";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH: Your GPU is too old and not supported by cuDNN";
    default:
      return "A call to cuDNN failed";
  }
}

#define CHECK_CUDNN(call)                                                      \
  do{                                                                              \
      const cudnnStatus_t error = call;                                         \
      if (error != CUDNN_STATUS_SUCCESS)                                        \
      {                                                                          \
          std::ostringstream sout;                                               \
          sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
          sout << "code: " << error << ", reason: " << cudnn_get_error_string(error);\
          throw dlib::cudnn_error(sout.str());                            \
      }                                                                          \
  }while(false)

#define CHECK_CUDA(call)                                                       \
  do{                                                                              \
      const cudaError_t error = call;                                            \
      if (error != cudaSuccess)                                                  \
      {                                                                          \
          std::ostringstream sout;                                               \
          sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
          sout << "code: " << error << ", reason: " << cudaGetErrorString(error);\
          throw dlib::cuda_error(sout.str());                                          \
      }                                                                          \
  }while(false)

namespace dlib {

  //dlib macros

  static cudnnTensorDescriptor_t descriptor(const dlib::tensor &t) {
    return (const cudnnTensorDescriptor_t) t.get_cudnn_tensor_descriptor().get_handle();
  }

  static cudnnTensorDescriptor_t descriptor(const dlib::cuda::tensor_descriptor &t) {
    return (const cudnnTensorDescriptor_t) t.get_handle();
  }

//RNNMode_t = CUDNN_LSTM or CUDNN_GRU or CUDNN_RNN_RELU or CUDNN_RNN_TANH
//direction mode = CUDNN_UNIDIRECTIONAL (front to end only) or CUDNN_BIDIRECTIONAL (separatly front to back and back to front)
//input mode = CUDNN_LINEAR_INPUT (matrix mult) or CUDNN_SKIP_INPUT (input needs to be the right size for the first layer)
//dropout descriptor - create with cudnnCreateDropoutDescriptor() and cudnnSetDropoutDescriptor()

//look at dlib cudnn handle stuff
//think about storing hidden state in memory or just not on the device?

  enum rnn_mode_t {
    RNN_RELU, RNN_TANH
  };
  enum direction_mode_t {
    UNIDIRECTIONAL, BIDIRECTIONAL
  };

//Assumes that we are going to use CUDNN - also instead of doing all this shit just do different classes for different RNN types...

  template<
          rnn_mode_t rnn_activation,
          direction_mode_t rnn_direction,
          int seq_length,      //Number of sequences to unroll over (roughly = num_inputs+num_outputs afaik)
          int rnn_hidden_size, //hidden state size - number of tensors - also fucks everything up if it's larer than 1...
          int rnn_num_layers,  //number of layers deep (at any time)
          int num_inputs,  //This is a number of tensors
          int num_outputs  //This is a number of tensors
          >
  class rnn_{

  private:
    dlib::resizable_tensor hx, hy;

    dlib::resizable_tensor w;
    cudnnFilterDescriptor_t wDesc;

    std::vector<dlib::resizable_tensor> x;
    std::vector<cudnnTensorDescriptor_t> xDescs;
    std::vector<dlib::resizable_tensor> y;
    std::vector<cudnnTensorDescriptor_t> yDescs;

    float *workspace;
    size_t workspace_size;

    float *training_reserve;
    size_t training_reserve_size;

    cudnnRNNDescriptor_t rnn_desc;
    cudnnDropoutDescriptor_t dropout_desc;
    cudnnHandle_t cudnn_handle;

  public:
    rnn_(){
      x.reserve(num_inputs);
      xDescs.reserve(num_inputs);
      y.reserve(num_outputs);
      yDescs.reserve(num_outputs);

      workspace_size = 0;
      training_reserve_size = 0;
    }

    template<typename SUBNET> //need potentially multiple subnets?
    void setup(const SUBNET &sub){

      DLIB_CASSERT(seq_length >= num_inputs + num_outputs, "Need seq_length >= num_inputs + num_outputs!");

      CHECK_CUDNN(cudnnCreate(&cudnn_handle));

      CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout_desc)); //maybe later implement dropout
      CHECK_CUDNN(cudnnSetDropoutDescriptor(dropout_desc,
                                            cudnn_handle,
                                            0.f,
                                            NULL,
                                            0,
                                            0));

      CHECK_CUDNN(cudnnCreateRNNDescriptor(&rnn_desc));

      cudnnRNNMode_t mde;
      cudnnDirectionMode_t dir;

      switch (rnn_activation) {
        case RNN_RELU:
          mde = CUDNN_RNN_RELU;
          break;
        case RNN_TANH:
          mde = CUDNN_RNN_TANH;
          break;
      }

      switch (rnn_direction) {
        case UNIDIRECTIONAL:
          dir = CUDNN_UNIDIRECTIONAL;
          hx.set_size(rnn_num_layers, x[0].k(), rnn_hidden_size, x[0].nr()); //check the last parameter, it might be wrong
          hy.set_size(rnn_num_layers, x[0].k(), rnn_hidden_size, x[0].nr());
          break;
        case BIDIRECTIONAL:
          dir = CUDNN_BIDIRECTIONAL;
          hx.set_size(rnn_num_layers, x[0].k(), 2*rnn_hidden_size, x[0].nr());
          hy.set_size(rnn_num_layers, x[0].k(), 2*rnn_hidden_size, x[0].nr());
          break;
      }

      CHECK_CUDNN(cudnnSetRNNDescriptor(rnn_desc,
                                        rnn_hidden_size,
                                        rnn_num_layers,
                                        dropout_desc,
                                        CUDNN_LINEAR_INPUT,
                                        dir,
                                        mde,
                                        CUDNN_DATA_FLOAT));

      DLIB_CASSERT(sub.get_output().k() == 1 && sub.get_output().nc() == 1, "Only column vectors!");

      int dims_x[3] = {(int)sub.get_output().num_samples(), (int)sub.get_output().nr(), 1}; //assert that it's a column vector
      int stride_x[3] = {dims_x[2]*dims_x[1], dims_x[2], 1};
      cudnnTensorDescriptor_t x_desc;
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
      CHECK_CUDNN(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 3, dims_x, stride_x));
      for (int i = 0; i < num_inputs; i++){ //this is messed up I think
        xDescs.push_back(x_desc);
      }

      int dims_y[3] = {(int)sub.get_output().num_samples(), (int)rnn_hidden_size*(1+(int)(rnn_direction == BIDIRECTIONAL)), 1};
      int stride_y[3] = {dims_y[2]*dims_y[1], dims_y[2], 1};
      cudnnTensorDescriptor_t y_desc;
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
      CHECK_CUDNN(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 3, dims_y, stride_y));
      for (int i = 0; i < num_outputs; i++){
        yDescs.push_back(y_desc);
      }

      //Set up workspace - this breaks, either xDescs are bad or rnn_desc is bad. My guess is it's xDescs somehow...? use torch impl
      CHECK_CUDNN(cudnnGetRNNWorkspaceSize(cudnn_handle,
                                           rnn_desc,
                                           seq_length,
                                           xDescs.data(),
                                           &workspace_size));

      CHECK_CUDA(cudaMallocManaged(&workspace, workspace_size));

      //Set up training reserve - only do this on the backward pass?
      CHECK_CUDNN(cudnnGetRNNTrainingReserveSize(cudnn_handle,
                                                 rnn_desc,
                                                 seq_length,
                                                 xDescs.data(),
                                                 &training_reserve_size));

      CHECK_CUDA(cudaMallocManaged(&training_reserve, training_reserve_size));

      tt::tensor_rand r (time(0));

      std::size_t params_size;
      CHECK_CUDNN(cudnnGetRNNParamsSize(cudnn_handle, rnn_desc, xDescs[0], &params_size, CUDNN_DATA_FLOAT));
      w.set_size(params_size);
      r.fill_gaussian(w, 0.0, 0.1);

      int sizes[] = {static_cast<int>(params_size), 1, 1};

      CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
      CHECK_CUDNN(cudnnSetFilterNdDescriptor(wDesc,
                                             CUDNN_DATA_FLOAT,
                                             CUDNN_TENSOR_NCHW,
                                             sizeof(sizes)/sizeof(sizes[0]),
                                             sizes));
    }

    void forward_inplace(const dlib::tensor &in, dlib::tensor &out) {
      //implement inplace and just have hx = hy? Need to save memory hardcore. Make sure to store y inside the class too.
      CHECK_CUDNN(cudnnRNNForwardTraining(cudnn_handle,
                                          rnn_desc,
                                          seq_length,
                                          xDescs.data(), in.device(),
                                          descriptor(hx), hx.device(),
                                          NULL, NULL,
                                          wDesc, w.device(),
                                          yDescs.data(), out.device(),
                                          descriptor(hy), hy.device(),
                                          NULL, NULL,
                                          workspace, workspace_size,
                                          training_reserve, training_reserve_size));

    }

  };

  template< //use the branching structure for things like inception-net to get more inputs...
          rnn_mode_t rnn_activation,
          direction_mode_t rnn_direction,
          int seq_length,      //Number of sequences to unroll over
          int rnn_hidden_size, //hidden state size (input/output dimensions don't matter, can have any number of inputs and outputs)
          int rnn_num_layers,   //number of layers deep (at any time)
//          int num_inputs,
//          int num_outputs,
          typename SUBNET //learn parameter packs
          >
  using rnn = add_layer<rnn_<rnn_activation, rnn_direction, seq_length, rnn_hidden_size, rnn_num_layers, 1, 1>, SUBNET>;

//  template<
//          direction_mode_t rnn_direction,
//          int seq_length,      //Number of sequences to unroll over (roughly = num_inputs+num_outputs afaik)
//          int rnn_hidden_size, //hidden state size - number of tensors - also fucks everything up if it's larer than 1...
//          int rnn_num_layers,  //number of layers deep (at any time)
//          int num_inputs = 1,  //This is a number of tensors
//          int num_outputs = 1  //This is a number of tensors
//  >
//  class gru_{
//
//  };

//  template<
//          direction_mode_t rnn_direction,
//          int seq_length,      //Number of sequences to unroll over (roughly = num_inputs+num_outputs afaik)
//          int rnn_hidden_size, //hidden state size - number of tensors - also fucks everything up if it's larer than 1...
//          int rnn_num_layers,  //number of layers deep (at any time)
//          int num_inputs = 1,  //This is a number of tensors
//          int num_outputs = 1  //This is a number of tensors
//  >
//  class lstm_{
//
//  };


//  template<
//        rnn_mode_t rnn_mode, //->change this so that there's like an rnn_ class (relu or tanh), lstm_ class, gru_ class...
//        direction_mode_t rnn_direction,
//        int seq_length,      //Number of sequences to unroll over (roughly = num_inputs+num_outputs afaik)
//        int rnn_hidden_size, //hidden state size - number of tensors - also fucks everything up if it's larer than 1...
//        int rnn_num_layers,  //number of layers deep (at any time)
//        int num_inputs = 1,  //This is a number of tensors
//        int num_outputs = 1  //This is a number of tensors
//        >
//  class rnn_ {
//
//  private:
//    dlib::resizable_tensor hx, hy;
//    dlib::resizable_tensor cx, cy;
//
//    float *w;
//    cudnnFilterDescriptor_t wDesc;
//
//    dlib::resizable_tensor x;
//    cudnnTensorDescriptor_t xDescs[num_inputs];
//    dlib::resizable_tensor y;
//    cudnnTensorDescriptor_t yDescs[num_outputs];
//
//    float *workspace;
//    size_t workspace_size;
//
//    float *training_reserve;
//    size_t training_reserve_size;
//
//    cudnnRNNDescriptor_t rnn_desc
// ;
//    cudnnDropoutDescriptor_t dropout_desc;
//    cudnnHandle_t cudnn_handle;
//
//
//  public:
//    rnn_() {}
//
//    ~rnn_(){
//      cudaFree(workspace);
//      cudaFree(training_reserve);
//    }
//
//    template<typename SUBNET>
//    void setup(const SUBNET &sub) {
//
////      DLIB_CASSERT(num_inputs == num_outputs);
//      DLIB_CASSERT(seq_length == num_inputs+num_outputs);
//
//      CHECK_CUDNN(cudnnCreate(&cudnn_handle));
//      CHECK_CUDNN(cudnnCreateRNNDescriptor(&rnn_desc
// ));
//      CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout_desc));
//      CHECK_CUDNN(cudnnSetDropoutDescriptor(dropout_desc,
//                                            cudnn_handle,
//                                            0.f,
//                                            nullptr,
//                                            0,
//                                            0));
//
//      x = sub.get_output();
//      DLIB_CASSERT(x.nc() == 1, "Only works with 1-D inputs!");
//      y.set_size(x.num_samples(), rnn_hidden_size, x.nr(), x.nc());
//
//      for (int i = 0; i < num_inputs; i++){
//        xDescs[i] = descriptor(x);
////        yDescs[i] = descriptor(y);
//      }
//
//      for (int i = 0; i < num_outputs; i++){
//        yDescs[i] = descriptor(y);
//      }
//
//      cudnnRNNMode_t mde;
//      cudnnDirectionMode_t dir;
//
//      switch (rnn_mode) {
//        case RNN_RELU:
//          mde = CUDNN_RNN_RELU;
//          break;
//        case RNN_TANH:
//          mde = CUDNN_RNN_TANH;
//          break;
//        case LSTM:
//          mde = CUDNN_LSTM;
//          break;
//        case GRU:
//          mde = CUDNN_GRU;
//          break;
//      }
//
//      switch (rnn_direction) {
//        case UNIDIRECTIONAL:
//          dir = CUDNN_UNIDIRECTIONAL;
//          break;
//        case BIDIRECTIONAL:
//          dir = CUDNN_BIDIRECTIONAL;
//          break;
//      }
//
//      CHECK_CUDNN(cudnnSetRNNDescriptor(rnn_desc
// ,
//                                        rnn_hidden_size,
//                                        rnn_num_layers,
//                                        dropout_desc,
//                                        CUDNN_LINEAR_INPUT,
//                                        dir,
//                                        mde,
//                                        CUDNN_DATA_FLOAT));
//
//      std::size_t params_size;
//      CHECK_CUDNN(cudnnGetRNNParamsSize(cudnn_handle, rnn_desc
// , xDescs[0], &params_size, CUDNN_DATA_FLOAT));
//
////      std::cout << params_size/sizeof(float) << std::endl;
//
//      CHECK_CUDA(cudaMallocManaged(&w, params_size));
//      int sizes[] = {static_cast<int>(params_size), 1, 1};
//
//      CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
//      CHECK_CUDNN(cudnnSetFilterNdDescriptor(wDesc,
//                                             CUDNN_DATA_FLOAT,
//                                             CUDNN_TENSOR_NCHW,
//                                             sizeof(sizes)/sizeof(sizes[0]),
//                                             sizes));
//
//        dlib::tt::tensor_rand r(time(0));
//
//      //Set up h
//      hx.set_size(rnn_num_layers, x.k(), rnn_hidden_size, x.nr());
//      hy.set_size(rnn_num_layers, x.k(), rnn_hidden_size, x.nr());
//                    //not sure about the last parameter, also third parameter should be doubled if bidirectional
////      r.fill_gaussian(hx, 0, 0.1);
//
//      //Set up c - the last parameter is wrong? (the others are dictated 100% by CUDA...)
//      cx.set_size(rnn_num_layers, x.k(), rnn_hidden_size, rnn_hidden_size);
//      cy.set_size(rnn_num_layers, x.k(), rnn_hidden_size, rnn_hidden_size);
//                    //worry about the last parameter and maybe also the doubling of the last parameter
//
//      //Set up workspace
//      CHECK_CUDNN(cudnnGetRNNWorkspaceSize(cudnn_handle,
//                                           rnn_desc
// ,
//                                           seq_length,
//                                           xDescs,
//                                           &workspace_size));
//
//      CHECK_CUDA(cudaMallocManaged(&workspace, workspace_size));
//
//      //Set up training reserve - only do this on the backward pass?
//      CHECK_CUDNN(cudnnGetRNNTrainingReserveSize(cudnn_handle,
//                                                 rnn_desc
// ,
//                                                 seq_length,
//                                                 xDescs,
//                                                 &training_reserve_size));
//
//      CHECK_CUDA(cudaMallocManaged(&training_reserve, training_reserve_size));
//
//    }
//
//    void forward_inplace(const dlib::tensor &in, dlib::tensor &out) {
//      //implement inplace and just have hx = hy? Need to save memory hardcore. Make sure to store y inside the class too.
//      CHECK_CUDNN(cudnnRNNForwardTraining(cudnn_handle, //also getting execution failed.......
//                                          rnn_desc
// ,
//                                          seq_length,
//                                          xDescs, in.device(),
//                                          descriptor(hx), hx.device(),
//                                          descriptor(cx), cx.device(),
//                                          wDesc, w,
//                                          yDescs, out.device(),
//                                          descriptor(hy), hy.device(),
//                                          descriptor(cy), cy.device(),
//                                          workspace, workspace_size,
//                                          training_reserve, training_reserve_size));
//
//    }
//
//    void backward_inplace(
//            const dlib::tensor &computed_output, // this parameter is optional - should I compute the output in backward?
//            const dlib::tensor &gradient_input,
//            dlib::tensor &data_grad,
//            dlib::tensor &params_grad) {
//      ;
//    }
//
//    const dlib::tensor &get_layer_params() const {
//      ; //cudnn get weights
//    }
//
//    dlib::tensor &get_layer_params() {
//      ;
//    }
//
//  };
//
//
//  template<
//          rnn_mode_t rnn_mode,
//          direction_mode_t rnn_direction,
//          int seq_length,      //Number of sequences to unroll over
//          int rnn_hidden_size, //hidden state size (input/output dimensions don't matter, can have any number of inputs and outputs)
//          int rnn_num_layers,   //number of layers deep (at any time)
//          int num_inputs,
//          int num_outputs,
//          typename SUBNET
//          >
//  using rnn = add_layer<
//          rnn_<rnn_mode, rnn_direction, seq_length, rnn_hidden_size, rnn_num_layers, num_inputs, num_outputs>,
//          SUBNET>;

}

#endif //HOMEAUTOMATION_CUDNN_RNN_H_HPP
