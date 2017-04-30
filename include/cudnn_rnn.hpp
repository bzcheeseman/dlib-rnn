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

#ifndef DLIB_USE_CUDA
#define DLIB_USE_CUDA

//dlib dependencies
#include <dlib/dnn.h>
#include <dlib/dnn/tensor.h>
#include <dlib/dnn/cuda_errors.h>
#include <dlib/dnn/cuda_dlib.h>
#include <dlib/dnn/cuda_utils.h>
#include <dlib/dnn/cudnn_dlibapi.h>
#include <dlib/dnn/layers.h>
#include <dlib/serialize.h>

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

  //accepts inputs of size (seq_length*minibatch, nr, nc) (the samples can repeat)
  template<int seq_length>
  class input_rnn {
    int seq_len;
  public:
    input_rnn():seq_len(seq_length) {}

    typedef matrix<unsigned char> input_type;

    template<typename forward_iterator>
    void to_tensor (
            forward_iterator ibegin,
            forward_iterator iend,
            resizable_tensor& data
    ) const
    {
      long minibatch = std::distance(ibegin,iend)/seq_length;
      data.set_size(minibatch, seq_length, ibegin->nr(), ibegin->nc());


      for(int n = 0; n < minibatch; n++){
        for (int k = 0; k < seq_length; k++){
          for (int nr = 0; nr < ibegin->nr(); nr++){
            for (int nc = 0; nc < ibegin->nc(); nc++){
              data.host_write_only()[((n*seq_length + k)*ibegin->nr() + nr)*ibegin->nc() + nc]
                      = (*(ibegin+(n*seq_length + k)))(nr,nc);
            }
          }
        }
      }

    }

    friend void serialize(const input_rnn &item, std::ostream &out){
      serialize(item.seq_len, out);
    }

    friend void deserialize(input_rnn &item, std::istream &in){
      deserialize(item.seq_len, in);
    }

  };

  enum rnn_mode_t {
    RNN_RELU, RNN_TANH, GRU
  };
  enum rnn_direction_t {
    UNIDIRECTIONAL, BIDIRECTIONAL
  };

  //Accepts inputs of dimension (seq_length, batch_size, input_size)
  template<
          rnn_mode_t rnn_activation,
          rnn_direction_t rnn_direction,
          int rnn_hidden_size, //hidden state size - number of tensors (needs to be comparable to the number of inputs?)
          int rnn_num_layers  //number of layers deep (at any time)
  >
  class rnn_ {

  private:
    dlib::resizable_tensor hx, hy, dhx, dhy;
    cudnnTensorDescriptor_t hx_desc;
    cudnnTensorDescriptor_t hy_desc;

    dlib::resizable_tensor w;
    cudnnFilterDescriptor_t wDesc;

    std::vector<cudnnTensorDescriptor_t> xDescs;
    resizable_tensor y;
    std::vector<cudnnTensorDescriptor_t> yDescs;

    cudnnRNNMode_t mde;
    cudnnDirectionMode_t dir;

    float *workspace;
    size_t workspace_size;

    float *training_reserve;
    size_t training_reserve_size;

    cudnnRNNDescriptor_t rnn_desc;
    cudnnDropoutDescriptor_t dropout_desc;
    cudnnHandle_t cudnn_handle;

    double learning_rate_multiplier;
    double weight_decay_multiplier;

    int batch_size;
    int seq_length;

    // to swap dimensions (a.k.a. transpose) just set_size(k, n, r, c)

    void transpose_nk(resizable_tensor &data){

      long old_n = data.num_samples();
      long old_k = data.k();

      data.set_size(old_k, old_n, data.nr(), data.nc());

    }

  public:
    rnn_() {
      workspace_size = 0;
      training_reserve_size = 0;
    }

//    rnn_(const rnn_ &other){
//      ;//todo
//    }

    double get_learning_rate_multiplier() const { return learning_rate_multiplier; }

    double get_weight_decay_multiplier() const { return weight_decay_multiplier; }

    void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }

    void set_weight_decay_multiplier(double val) { weight_decay_multiplier = val; }

    const tensor &get_layer_params() const { return w; }

    tensor &get_layer_params() { return w; }

    rnn_mode_t get_mode() { return rnn_activation; }

    rnn_direction_t get_direction() { return rnn_direction; }

    template<typename SUBNET>
    //need potentially multiple subnets?
    void setup(const SUBNET &sub) {

      //minibatch samples go into each sequence input! That's why we want that one first.
      batch_size = (int)sub.get_output().num_samples();
      seq_length = (int)sub.get_output().k();

      CHECK_CUDNN(cudnnCreate(&cudnn_handle));

      CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout_desc)); //maybe later implement dropout
      CHECK_CUDNN(cudnnSetDropoutDescriptor(dropout_desc,
                                            cudnn_handle,
                                            0.f,
                                            NULL,
                                            0,
                                            0));

      CHECK_CUDNN(cudnnCreateRNNDescriptor(&rnn_desc));

      switch (rnn_activation) {
        case RNN_RELU:
          mde = CUDNN_RNN_RELU;
          break;
        case RNN_TANH:
          mde = CUDNN_RNN_TANH;
          break;
        case GRU:
          mde = CUDNN_GRU;
          break;
      }

      int direction;
      switch (rnn_direction) {
        case UNIDIRECTIONAL:
          dir = CUDNN_UNIDIRECTIONAL;
          direction = 1;
          break;
        case BIDIRECTIONAL:
          dir = CUDNN_BIDIRECTIONAL;
          direction = 2;
          break;
      }

      int dims_x[3] = {batch_size, (int) sub.get_output().nr(), (int) sub.get_output().nc()};
      int stride_x[3] = {dims_x[2] * dims_x[1], dims_x[2], 1};
      cudnnTensorDescriptor_t x_desc;
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
      CHECK_CUDNN(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 3, dims_x, stride_x));
      for (int i = 0; i < seq_length; i++) {
        xDescs.push_back(x_desc);
      }

      int dims_y[3] = {batch_size, rnn_hidden_size * direction, 1};
      int stride_y[3] = {dims_y[2] * dims_y[1], dims_y[2], 1};
      cudnnTensorDescriptor_t y_desc;
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
      CHECK_CUDNN(cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 3, dims_y, stride_y));
      y.set_size(dims_y[0], dims_y[1], dims_y[2]);
      for (int i = 0; i < seq_length; i++) {
        yDescs.push_back(y_desc);
      }

      tt::tensor_rand r(time(0));

      int dim_h[3] = {rnn_num_layers * direction, batch_size, rnn_hidden_size};
      int stride_h[3] = {dim_h[2] * dim_h[1], dim_h[2], 1};

      CHECK_CUDNN(cudnnCreateTensorDescriptor(&hx_desc));
      CHECK_CUDNN(cudnnCreateTensorDescriptor(&hy_desc));

      CHECK_CUDNN(cudnnSetTensorNdDescriptor(hx_desc, CUDNN_DATA_FLOAT, 3, dim_h, stride_h));
      CHECK_CUDNN(cudnnSetTensorNdDescriptor(hy_desc, CUDNN_DATA_FLOAT, 3, dim_h, stride_h));

      hx.set_size(dim_h[0], dim_h[1], dim_h[2] + (rnn_hidden_size % 2));
      hy.set_size(dim_h[0], dim_h[1], dim_h[2] + (rnn_hidden_size % 2));

      dhy.copy_size(hy);
      dhx.copy_size(hx);

      dhy = 0.0f;
      dhx = 0.0f;

      r.fill_gaussian(hx, 0, 0.1);


      CHECK_CUDNN(cudnnSetRNNDescriptor(rnn_desc,
                                        rnn_hidden_size,
                                        rnn_num_layers,
                                        dropout_desc,
                                        CUDNN_LINEAR_INPUT,
                                        dir,
                                        mde,
                                        CUDNN_DATA_FLOAT));

      //Set up workspace - this breaks, either xDescs are bad or rnn_desc is bad. My guess is it's xDescs somehow...? use torch impl
      CHECK_CUDNN(cudnnGetRNNWorkspaceSize(cudnn_handle,
                                           rnn_desc,
                                           seq_length,
                                           xDescs.data(),
                                           &workspace_size));

      CHECK_CUDA(cudaMallocManaged(&workspace, workspace_size * sizeof(float)));

      //Set up training reserve - only do this on the backward pass?
      CHECK_CUDNN(cudnnGetRNNTrainingReserveSize(cudnn_handle,
                                                 rnn_desc,
                                                 seq_length,
                                                 xDescs.data(),
                                                 &training_reserve_size));

      CHECK_CUDA(cudaMallocManaged(&training_reserve, training_reserve_size));

      std::size_t params_size;
      CHECK_CUDNN(cudnnGetRNNParamsSize(cudnn_handle, rnn_desc, xDescs[0], &params_size, CUDNN_DATA_FLOAT));
      w.set_size(params_size / sizeof(float));
      r.fill_gaussian(w, 0.0, 0.1);

      int sizes[] = {static_cast<int>(params_size), 1, 1};

      CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
      CHECK_CUDNN(cudnnSetFilterNdDescriptor(wDesc,
                                             CUDNN_DATA_FLOAT,
                                             CUDNN_TENSOR_NCHW,
                                             sizeof(sizes) / sizeof(sizes[0]),
                                             sizes));

    }

    template<typename SUBNET>
    void forward(const SUBNET &sub, resizable_tensor &out) {  // This should work, need to check when cuda works again

      resizable_tensor &input = sub.get_output(); //rename the input to the rnn
      transpose_nk(input); // transpose the tensor, get k along the first dimension

      CHECK_CUDNN(cudnnRNNForwardTraining(cudnn_handle,
                                          rnn_desc,
                                          seq_length,
                                          xDescs.data(), input.device(),
                                          hx_desc, hx.device(),
                                          NULL, NULL,
                                          wDesc, w.device(),
                                          yDescs.data(), y.device(),
                                          hy_desc, hy.device(),
                                          NULL, NULL,
                                          workspace, workspace_size,
                                          training_reserve, training_reserve_size));

      hx = hy; //update hidden state
      hy = 0; //reset output hidden to zero

      out = y;
      transpose_nk(out); // re-transpose to get the batch along the first dimension

    }

    template<typename SUBNET>
    void backward(const tensor &gradient_input, // from previous layer
                  SUBNET &sub,  // closer to data (need to feed stuff to the subnet)
                  tensor &params_grad)
    // output, params_grad == grad(dot(computed_output (here our stored y), gradient_input), get_layer_params())
    {

      // layer<I>(sub).get_gradient_input() += data_gradient_I for all valid I
      // with data_gradient_I = gradient of layer I wrt gradient_input (dot(layer<I>(sub).get_output(), gradient_input)

      std::cout << gradient_input.num_samples() << " " << gradient_input.k() << " " << gradient_input.nr() << " " << gradient_input.nc() << std::endl;
      std::cout << sub.get_output().num_samples() << " " << sub.get_output().k() << " " << sub.get_output().nr() << " " << sub.get_output().nc() << std::endl;
      std::cout << sub.get_gradient_input().num_samples() << " " << sub.get_gradient_input().k() << " " << sub.get_gradient_input().nr() << " " << sub.get_gradient_input().nc() << std::endl;

      resizable_tensor grad_in = transpose_nk(gradient_input);
      resizable_tensor sub_out = transpose_nk(sub.get_output());
      resizable_tensor sub_grad_in = transpose_nk(sub.get_gradient_input());

      std::cout << grad_in.num_samples() << " " << grad_in.k() << " " << grad_in.nr() << " " << grad_in.nc() << std::endl;
      std::cout << sub_out.num_samples() << " " << sub_out.k() << " " << sub_out.nr() << " " << sub_out.nc() << std::endl;
      std::cout << sub_grad_in.num_samples() << " " << sub_grad_in.k() << " " << sub_grad_in.nr() << " " << sub_grad_in.nc() << std::endl;

      CHECK_CUDNN(cudnnRNNBackwardData(cudnn_handle,
                                       rnn_desc,
                                       seq_length,
                                       yDescs.data(), y.device(),
                                       yDescs.data(), grad_in.device(),
                                       hy_desc, dhy.device(),
                                       NULL, NULL, //dcy
                                       wDesc, w.device(),
                                       hx_desc, hx.device(),
                                       NULL, NULL, //cx
                                       xDescs.data(), sub_grad_in.device(),
                                       hx_desc, dhx.device(),
                                       NULL, NULL, //dcx
                                       workspace, workspace_size,
                                       training_reserve, training_reserve_size));

      if (learning_rate_multiplier != 0) {
        CHECK_CUDNN(cudnnRNNBackwardWeights(cudnn_handle,
                                            rnn_desc,
                                            seq_length,
                                            xDescs.data(), sub_out.device(),
                                            hx_desc, hx.device(),
                                            yDescs.data(), y.device(),
                                            workspace, workspace_size,
                                            wDesc, params_grad.device(),
                                            training_reserve, training_reserve_size));
      }

      //clip output dhx to avoid huge gradients
      tt::threshold(dhx, 5.f); //clip upper
      tt::threshold(dhx, -5.f); //clip lower
      dhy = dhx;
      tt::add(1.0, hx, (float)learning_rate_multiplier, dhx);

//      params_grad = transpose_kn(params_grad); //didn't work

//      sub.get_output() = transpose_kn(sub_out);
//      sub.get_gradient_input() = transpose_kn(sub_grad_in);

    }

    friend std::ostream &operator<<(std::ostream &out, const rnn_ &item) {
      out << "rnn ";

      switch (rnn_activation) {
        case RNN_RELU:
          out << "RELU ";
          break;
        case RNN_TANH:
          out << "TANH ";
          break;
        case GRU:
          out << "GRU ";
          break;
      }

      switch (rnn_direction) {
        case UNIDIRECTIONAL:
          out << "UNIDIRECTIONAL ";
          break;
        case BIDIRECTIONAL:
          out << "BIDIRECTIONAL ";
          break;
      }

      out << " learning_rate_mult=" << item.learning_rate_multiplier;
      out << " weight_decay_mult=" << item.weight_decay_multiplier;

      return out;
    }

    friend void serialize(const rnn_ &item, std::ostream &out){
      serialize(item.seq_length, out);
      serialize(item.workspace_size, out);
      serialize(item.weight_decay_multiplier, out);
      serialize(item.training_reserve_size, out);
      serialize(item.w, out);
      serialize(item.dhx, out);
      serialize(item.dhy, out);
      serialize(item.hx, out);
      serialize(item.hy, out);
      serialize(item.y, out);
    }

    friend void deserialize(rnn_ &item, std::istream &in){
      deserialize(item.seq_length, in);
      deserialize(item.workspace_size, in);
      CHECK_CUDA(cudaMallocManaged(&(item.workspace), item.workspace_size * sizeof(float)));
      deserialize(item.weight_decay_multiplier, in);
      deserialize(item.training_reserve_size, in);
      CHECK_CUDA(cudaMallocManaged(&(item.training_reserve), item.training_reserve_size));
      deserialize(item.w, in);
      deserialize(item.dhx, in);
      deserialize(item.dhy, in);
      deserialize(item.hx, in);
      deserialize(item.hy, in);
      deserialize(item.y, in);
    }

  };

  template< //use the branching structure for things like inception-net to get more inputs...
          rnn_mode_t rnn_activation,
          rnn_direction_t rnn_direction,
          int rnn_hidden_size, //hidden state size (input/output dimensions don't matter, can have any number of inputs and outputs)
          int rnn_num_layers,  //number of layers deep (at any time)
          typename SUBNET //learn parameter packs
  >
  using rnn = add_layer<rnn_<rnn_activation, rnn_direction, rnn_hidden_size, rnn_num_layers>, SUBNET>;

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
}

#endif //DLIB_USE_CUDA
#endif //HOMEAUTOMATION_CUDNN_RNN_H_HPP
