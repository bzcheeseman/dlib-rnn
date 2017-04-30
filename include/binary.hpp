//
// Created by Aman LaChapelle on 2/1/17.
//
// dlib_rnn
// Copyright (c) 2017 Aman LaChapelle
// Full license at dlib_rnn/LICENSE.txt
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


#ifndef DLIB_RNN_BINARY_HPP
#define DLIB_RNN_BINARY_HPP

#include <bitset>
#include <cassert>
#include <cmath>
#include <random>
#include <memory>
#include <functional>
#include <dlib/dnn.h>
//#include <dlib/dnn/cuda_utils.h>
//#include <dlib/dnn/gpu_data.h>

#include <unsupported/Eigen/CXX11/Tensor>

namespace dlib
{

  class bin_tensor {
    const size_t size;
    size_t N, K, NR, NC;
    std::bitset<size> storage;

  public:

    bin_tensor(size_t N, size_t K, size_t NR, size_t NC): size(N*K*NR*NC), N(N), K(K), NR(NR), NC(NC) {}

    bin_tensor operator()(); //slicing returns another bin_tensor
    //compute bit masks for access operations (like slicing)
  };

//  template<size_t N=1, size_t K=1, size_t NR=1, size_t NC=1>
//  struct bin_tensor {
//    std::bitset<N*K*NR*NC> storage;
//
//    size_t num_samples() { return N; }
//    size_t k() { return K; }
//    size_t nr() { return NR; }
//    size_t nc() { return NC; }
//
//    //access operators - note that these are std::bitset::reference so use auto when accessing or create a new struct as a wrapper
//    //gemm kernel
//    //conv kernel - each depth of conv weights 3-tensor has the same weights and bias, N really is just number of samples
//    // - also look at im2col for conv layer and pool layer use darknet stuff - lots of good C code there
//    //activations
//    //binarize from dlib::tensor
//  };
//
//  struct bin_tensor {
//    std::vector<bool> storage;
//
//    size_t N, K, NR, NC;
//
//    size_t num_samples() { return N; }
//    size_t k() { return K; }
//    size_t nr() { return NR; }
//    size_t nc() { return NC; }
//
//    bin_tensor(size_t n=1, size_t k=1, size_t nr=1, size_t nc=1): storage(n*k*nr*nc), N(n), K(k) , NR(nr), NC(nc) {}
//  };


  namespace bin_helper {

    struct sign{
//      typedef float type;
      float operator()(float f){
        if (f >= 0) f = 1.f;
        else f = -1.f;
        return f;
      }
    };

//    auto sign(float &f){ //change these to TensorCwiseUnaryOp?
//      if (f >= 0) f = 1.f;
//      else f = -1.f;
//      return f;
//    }

    struct clip_input{
//      typedef float type;
      float operator()(float f){
        if (f >= 1.f) f = 1.f;
        if (f <= 0.f) f = 0.f;
        else ;
        return f;
      }
    };

//    auto clip_input(float &f){
//      if (f >= 1.f) f = 1.f;
//      if (f <= 0.f) f = 0.f;
//      else ;
//      return f;
//    }

    Eigen::Tensor<float, 4> toEigen(const dlib::resizable_tensor &t) {
      int tn = t.num_samples(), tk = t.k(), tnr = t.nr(), tnc = t.nc();

      Eigen::Tensor<float, 4> out(tn, tk, tnr, tnc);
      for (int i = 0; i < tn; i++) {
        for (int j = 0; j < tk; j++) {
          for (int k = 0; k < tnr; k++) {
            for (int l = 0; l < tnc; l++) {
              out(i, j, k, l) = t.host()[((i * tk + j) * tnr + k) * tnc + l];
            }
          }
        }
      }

      return out;
    }

    Eigen::Tensor<float, 4> toEigen(dlib::resizable_tensor &&t) {
      int tn = t.num_samples(), tk = t.k(), tnr = t.nr(), tnc = t.nc();

      Eigen::Tensor<float, 4> out(tn, tk, tnr, tnc);
      for (int i = 0; i < tn; i++) {
        for (int j = 0; j < tk; j++) {
          for (int k = 0; k < tnr; k++) {
            for (int l = 0; l < tnc; l++) {
              out(i, j, k, l) = t.host()[((i * tk + j) * tnr + k) * tnc + l];
            }
          }
        }
      }

      return out;
    }

    dlib::resizable_tensor toDlib(const Eigen::Tensor<float, 4> &t) {
      int tn = t.dimensions()[0], tk = t.dimensions()[1], tnr = t.dimensions()[2], tnc = t.dimensions()[3];

      dlib::resizable_tensor out(tn, tk, tnr, tnc);
      for (int i = 0; i < tn; i++) {
        for (int j = 0; j < tk; j++) {
          for (int k = 0; k < tnr; k++) {
            for (int l = 0; l < tnc; l++) {
              out.host()[((i * tk + j) * tnr + k) * tnc + l] = t(i, j, k, l);
            }
          }
        }
      }

      return out;
    }

    dlib::resizable_tensor toDlib(Eigen::Tensor<float, 4> &&t) {
      int tn = t.dimensions()[0], tk = t.dimensions()[1], tnr = t.dimensions()[2], tnc = t.dimensions()[3];

      dlib::resizable_tensor out(tn, tk, tnr, tnc);
      for (int i = 0; i < tn; i++) {
        for (int j = 0; j < tk; j++) {
          for (int k = 0; k < tnr; k++) {
            for (int l = 0; l < tnc; l++) {
              out.host()[((i * tk + j) * tnr + k) * tnc + l] = t(i, j, k, l);
            }
          }
        }
      }

      return out;
    }

    Eigen::Tensor<float,4> hard_sigmoid(Eigen::Tensor<float, 4> &t) { //in range [0,1]
      Eigen::Tensor<float, 4> ones(t.dimensions()), zeros(t.dimensions());
      ones.setConstant(1.f);
      zeros.setConstant(0.f);
      t = (0.5*(t+ones)).eval();
      t.unaryExpr(clip_input());
      return t;
    }

    Eigen::Tensor<float, 4> binary_tanh(Eigen::Tensor<float, 4> &t) { //in range [-1,1]
      return 2.f * hard_sigmoid(t) - 1.f;
    }

    Eigen::Tensor<bool, 4> binary_sigmoid(Eigen::Tensor<float, 4> &t) { //this is actually {0,1}
      hard_sigmoid(t);
      return t.cast<bool>();
    }

    Eigen::Tensor<bool, 4> binarize_filters(Eigen::Tensor<float, 4> &W, Eigen::Tensor<float, 1> &alpha) {
      W.unaryExpr(sign());
      alpha = W.abs().mean(Eigen::array<int, 3>{1,2,3});
      return W.cast<bool>();
    }

    Eigen::Tensor<bool, 4> binarize_filters(Eigen::Tensor<float, 4> &&W, Eigen::Tensor<float, 1> &alpha) {
      W.unaryExpr(sign());
      alpha = W.abs().mean(Eigen::array<int, 3>{1,2,3});
      return W.cast<bool>();
    }

    Eigen::Tensor<bool,4> binarize_input(Eigen::Tensor<float,4> &in, Eigen::Tensor<float,3> &A, Eigen::Tensor<float,2> k_filter){
      Eigen::array<ptrdiff_t, 2> dims({1,2});
      A.convolve(k_filter, dims);
      return binary_tanh(in).cast<bool>();
    }

    Eigen::Tensor<bool,4> binarize_input(Eigen::Tensor<float,4> &&in, Eigen::Tensor<float,3> &A, Eigen::Tensor<float,2> k_filter){
      Eigen::array<ptrdiff_t, 2> dims({1,2});
      A.convolve(k_filter, dims);
      return binary_tanh(in).cast<bool>();
    }


  };
};



namespace dlib
{
  template<
          long _num_filters,
          long _nr,
          long _nc,
          int _stride_y,
          int _stride_x,
          int _padding_y = _stride_y!= 1 ? 0 : _nr/2,
          int _padding_x = _stride_x!= 1 ? 0 : _nc/2
          >
  class binary_conv_
  {

    Eigen::Tensor<bool,4> *filters;
    Eigen::Tensor<bool,4> *most_recent;
    Eigen::Tensor<float,4> *W; //(num_samples, k, nr, nc)
    Eigen::Tensor<float,3> *K;
    Eigen::TensorFixedSize<float, Eigen::Sizes<_nr,_nc> > k;
    Eigen::Tensor<float,1> *alpha;
//    const Eigen::ThreadPoolDevice eigen_tp_device;

  public:
    binary_conv_(){}
    ~binary_conv_(){}

    template<typename SUBNET>
    void setup(
            const SUBNET &sub
    )
    {
      long num_inputs = _nr*_nc*sub.get_output().k();
      long num_outputs = _num_filters;

      const Eigen::array<long,4> filter_dims ({_num_filters, sub.get_output().k(), _nr, _nc});
      const Eigen::array<long,3> K_dims ({_num_filters, _nr, _nc});

      filters = new Eigen::Tensor<bool, 4> (filter_dims);
      W = new Eigen::Tensor<float, 4> (filter_dims);
      K = new Eigen::Tensor<float, 3> (K_dims);
      alpha = new Eigen::Tensor<float,1> (sub.get_output().num_samples());

      W->setRandom();

      k.setConstant(1.f/((float)_nr*_nc));
    }

    template<typename SUBNET>
    void forward(const SUBNET &sub, resizable_tensor &data_output){ //for forward - take mean along second dimension (k dimension)
      Eigen::Tensor<float,4> input = bin_helper::toEigen(sub.get_output());

      //Get the A tensor
      Eigen::array<ptrdiff_t, 1> dim({1});
      Eigen::Tensor<float,3> A = input.mean(dim);
      //Binarize the inputs
      Eigen::Tensor<bool,4> bin_input = bin_helper::binarize_input(input, A, k); //once this returns A will be K

      *K = A;

      //Binarize the filters
      *filters = bin_helper::binarize_filters(*W, *alpha);

      //Now we have K, alpha, binI and binW - we can do the convolution
      Eigen::array<int, 4> dims({0,1,2,3});
      bin_input.convolve(*filters, dims); //should change this

      //Broadcast K and alpha along their axes and then multiply coefficient-wise
      Eigen::array<int,3> bcast_alpha_dims({1,2,3});

      bin_input *= (*K).broadcast(Eigen::array<long,4>({0,1,0,0})); //this doesn't work, need to find a better way to broadcast
      bin_input *= (*alpha).broadcast(bcast_alpha_dims);

      *most_recent = bin_input;

      data_output = bin_helper::toDlib(bin_input.cast<float>());

    }

    template<typename SUBNET>
    void backward(const tensor& gradient_input,
                  SUBNET& sub,
                  tensor& params_grad){ //still need to compute the backwards pass
      ;
    }


  };

  template<long _num_filters,
          long _nr,
          long _nc,
          int _stride_y,
          int _stride_x,
          typename SUBNET
  >
  using binary_conv = add_layer<binary_conv_<_num_filters, _nr, _nc,
                                _stride_y, _stride_x>, SUBNET>;

}



//template<int N, int NR, int NC>
//struct binary_tensor{
//
//  typedef typename std::bitset<N*NR*NC>::reference bitref;
//
//  inline const int n(){ return N; }
//  inline const int nr(){ return NR; }
//  inline const int nc(){ return NC; }
//  inline const int size(){ return N*NR*NC; }
//
//  //constructor that initializes the tensor to be random
//
//  inline bitref operator()(int n, int nr, int nc){
//    return me[n + N*(nc + NC*nr)];
//  }
//
//  inline binary_tensor<N, NR, NC> operator+=(binary_tensor<N, NR, NC> &other){
//    for (int i = 0; i < N; i++){
//      for (int j = 0; j < NR; j++){
//        for (int k = 0; k < NC; k++){
//          this->operator()(i,j,k) = this->operator()(i,j,k) & other(i,j,k);
//        }
//      }
//    }
//    return *this;
//  }
//
//  inline binary_tensor<N, NR, NC> operator*=(binary_tensor<N, NR, NC> &other){
//    for (int i = 0; i < N; i++){
//      for (int j = 0; j < NR; j++){
//        for (int k = 0; k < NC; k++){
//          this->operator()(i,j,k) = this->operator()(i,j,k) ^ other(i,j,k);
//        }
//      }
//    }
//    return *this;
//  }
//
//  inline binary_tensor<N, NR, NC> operator~(){
//    for (int i = 0; i < N; i++){
//      for (int j = 0; j < NR; j++){
//        for (int k = 0; k < NC; k++){
//          this->operator()(i,j,k) = ~this->operator()(i,j,k);
//        }
//      }
//    }
//    return *this;
//  }
//
//private:
//  std::bitset<N*NR*NC> me;
//
//};
//
//template<int N, int NR, int NC>
//inline binary_tensor<N, NR, 1> operator,(binary_tensor<N, NR, NC> &first, binary_tensor<N, NC, 1> &other){
//
//  binary_tensor<N, NR, 1> out;
//  std::bitset<NR> col_new;
//  std::bitset<NC> row_me;
//  std::bitset<NC> col_other;
//
//  //parallelize with omp
//  for (int i = 0; i < N; i++){
//    for (int j = 0; j < NR; j++){
//      for (int k = 0; k < NC; k++){
//        row_me[k] = first(i,j,k);
//        col_other[k] = other(i,k,0);
//      }
//      row_me ^= col_other; //multiply the row/col (-1*-1 = 1, 1*-1 = -1, etc.)
//      col_new[j] = (~row_me).count();
//      out(i,j,0) = col_new[j]; //set output value = thing just calculated
//    }
//  }
//  return out;
//
//}
//
//template<int N, int NR, int NC>
//inline binary_tensor<N, NR+1, NC> concatenate(binary_tensor<N, NR, NC> &first, binary_tensor<N, 1, NC> &other){
//  binary_tensor<N, NR+1, NC> out;
//  for (int i = 0; i < N; i++){
//    for (int j = 0; j < NR+1; j++){
//      for (int k = 0; k < NC; k++){
//        if (j == NR){
//          out(i,j,k) = other(i,0,k);
//        }
//        else{
//          out(i,j,k) = first(i,j,k);
//        }
//      }
//    }
//  }
//  return out;
//}
//
//template<int N, int NR, int NC>
//inline binary_tensor<N, NR+1, NC> concatenate(binary_tensor<N, NR, NC> &&first, binary_tensor<N, 1, NC> &other){
//  binary_tensor<N, NR+1, NC> out;
//  for (int i = 0; i < N; i++){
//    for (int j = 0; j < NR+1; j++){
//      for (int k = 0; k < NC; k++){
//        if (j == NR){
//          out(i,j,k) = other(i,0,k);
//        }
//        else{
//          out(i,j,k) = first(i,j,k);
//        }
//      }
//    }
//  }
//  return out;
//}
//
//template<int N, int NR, int NC>
//inline binary_tensor<N, NR, NC> operator+(binary_tensor<N, NR, NC> &first, binary_tensor<N, NR, NC> &other){
//  binary_tensor<N, NR, NC> out;
//  for (int i = 0; i < N; i++){
//    for (int j = 0; j < NR; j++){
//      for (int k = 0; k < NC; k++){
//        out(i,j,k) = first(i,j,k) & other(i,j,k);
//      }
//    }
//  }
//  return out;
//}
//
//template<int N, int NR, int NC>
//inline binary_tensor<N, NR, NC> operator*(binary_tensor<N, NR, NC> &first, binary_tensor<N, NR, NC> &other){
//  binary_tensor<N, NR, NC> out;
//  for (int i = 0; i < N; i++){
//    for (int j = 0; j < NR; j++){
//      for (int k = 0; k < NC; k++){
//        out(i,j,k) = first(i,j,k) ^ other(i,j,k);
//      }
//    }
//  }
//  return out;
//}
//
//template<int N, int NR, int NC>
//inline binary_tensor<N, NR, NC> operator*(binary_tensor<N, NR, NC> &&first, binary_tensor<N, NR, NC> &other){
//  binary_tensor<N, NR, NC> out;
//  for (int i = 0; i < N; i++){
//    for (int j = 0; j < NR; j++){
//      for (int k = 0; k < NC; k++){
//        out(i,j,k) = first(i,j,k) ^ other(i,j,k);
//      }
//    }
//  }
//  return out;
//}
//
//template<int x_size, int h_n>
//class gru {
//  float alpha_z, alpha_r, alpha_h;
//  binary_tensor<x_size, h_n, h_n+1> B_z, B_r, B_h;
//  binary_tensor<x_size, h_n, 1> h, h_final, rt, zt, ht; //ht = h_twiddle
//  binary_tensor<x_size, h_n+1, 1> hx, rhx;
//
//public:
//  void setup(){
//    ; //random tensors for B's, random floats for alpha's
//  }
//
//  binary_tensor<x_size, h_n, 1> forward(binary_tensor<x_size,1,1> input){
//
////    std::cout << sizeof(binary_tensor<x_size, h_n, 1>) << " " << sizeof(Eigen::Tensor<bool,3>(x_size, h_n, 1)) << std::endl;
//
////    //first inputs
////    hx = concatenate(h, input);
////    zt = (B_z,hx);
////    rt = (B_r,hx);
////    rhx = concatenate(rt * h, input);
////    //now make the update for h
////    ht = (B_h,rhx) * zt;
////    ~zt;
////    h *= zt; //this guy has sigm(1-alpha_z)
////    h_final = h + ht;
////    alpha_h = _helper::sigm(1.f-alpha_z) + tanhf(alpha_h * _helper::sigm(alpha_r))*_helper::sigm(alpha_z); //update all the float stuff
////    return h_final;
//  }
//
//};

#endif //DLIB_RNN_BINARY_HPP
