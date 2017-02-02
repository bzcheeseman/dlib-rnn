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

#include <unsupported/Eigen/CXX11/Tensor>

namespace _helper {
  bool activ(float x){
    return x < 0;
  }

  bool randomBool() {
    std::mt19937 gen;
    std::bernoulli_distribution dist;
    return dist(gen);
  }
}

template<int N, int NR, int NC>
struct binary_tensor{

  typedef typename std::bitset<N*NR*NC>::reference bitref;

  inline const int n(){ return N; }
  inline const int nr(){ return NR; }
  inline const int nc(){ return NC; }
  inline const int size(){ return N*NR*NC; }

  //constructor that initializes the tensor to be random

  inline bitref operator()(int n, int nr, int nc){
    return me[n + N*(nc + NC*nr)];
  }

  inline binary_tensor<N, NR, NC> operator+=(binary_tensor<N, NR, NC> &other){
    for (int i = 0; i < N; i++){
      for (int j = 0; j < NR; j++){
        for (int k = 0; k < NC; k++){
          this->operator()(i,j,k) = this->operator()(i,j,k) & other(i,j,k);
        }
      }
    }
    return *this;
  }

  inline binary_tensor<N, NR, NC> operator*=(binary_tensor<N, NR, NC> &other){
    for (int i = 0; i < N; i++){
      for (int j = 0; j < NR; j++){
        for (int k = 0; k < NC; k++){
          this->operator()(i,j,k) = this->operator()(i,j,k) ^ other(i,j,k);
        }
      }
    }
    return *this;
  }

  inline binary_tensor<N, NR, NC> operator~(){
    for (int i = 0; i < N; i++){
      for (int j = 0; j < NR; j++){
        for (int k = 0; k < NC; k++){
          this->operator()(i,j,k) = ~this->operator()(i,j,k);
        }
      }
    }
    return *this;
  }

private:
  std::bitset<N*NR*NC> me;

};

template<int N, int NR, int NC>
inline binary_tensor<N, NR, 1> operator,(binary_tensor<N, NR, NC> &first, binary_tensor<N, NC, 1> &other){

  binary_tensor<N, NR, 1> out;
  std::bitset<NR> col_new;
  std::bitset<NC> row_me;
  std::bitset<NC> col_other;

  //parallelize with omp
  for (int i = 0; i < N; i++){
    for (int j = 0; j < NR; j++){
      for (int k = 0; k < NC; k++){
        row_me[k] = first(i,j,k);
        col_other[k] = other(i,k,0);
      }
      row_me ^= col_other; //multiply the row/col (-1*-1 = 1, 1*-1 = -1, etc.)
      col_new[j] = (~row_me).count()%2; //counts the number of 1's and mod2 to binarize
      out(i,j,0) = col_new[j]; //set output value = thing just calculated
    }
  }
  return out;

}

template<int N, int NR, int NC>
inline binary_tensor<N, NR+1, NC> concatenate(binary_tensor<N, NR, NC> &first, binary_tensor<N, 1, NC> &other){
  binary_tensor<N, NR+1, NC> out;
  for (int i = 0; i < N; i++){
    for (int j = 0; j < NR+1; j++){
      for (int k = 0; k < NC; k++){
        if (j == NR){
          out(i,j,k) = other(i,0,k);
        }
        else{
          out(i,j,k) = first(i,j,k);
        }
      }
    }
  }
  return out;
}

template<int N, int NR, int NC>
inline binary_tensor<N, NR+1, NC> concatenate(binary_tensor<N, NR, NC> &&first, binary_tensor<N, 1, NC> &other){
  binary_tensor<N, NR+1, NC> out;
  for (int i = 0; i < N; i++){
    for (int j = 0; j < NR+1; j++){
      for (int k = 0; k < NC; k++){
        if (j == NR){
          out(i,j,k) = other(i,0,k);
        }
        else{
          out(i,j,k) = first(i,j,k);
        }
      }
    }
  }
  return out;
}

template<int N, int NR, int NC>
inline binary_tensor<N, NR, NC> operator+(binary_tensor<N, NR, NC> &first, binary_tensor<N, NR, NC> &other){
  binary_tensor<N, NR, NC> out;
  for (int i = 0; i < N; i++){
    for (int j = 0; j < NR; j++){
      for (int k = 0; k < NC; k++){
        out(i,j,k) = first(i,j,k) & other(i,j,k);
      }
    }
  }
  return out;
}

template<int N, int NR, int NC>
inline binary_tensor<N, NR, NC> operator*(binary_tensor<N, NR, NC> &first, binary_tensor<N, NR, NC> &other){
  binary_tensor<N, NR, NC> out;
  for (int i = 0; i < N; i++){
    for (int j = 0; j < NR; j++){
      for (int k = 0; k < NC; k++){
        out(i,j,k) = first(i,j,k) ^ other(i,j,k);
      }
    }
  }
  return out;
}

template<int N, int NR, int NC>
inline binary_tensor<N, NR, NC> operator*(binary_tensor<N, NR, NC> &&first, binary_tensor<N, NR, NC> &other){
  binary_tensor<N, NR, NC> out;
  for (int i = 0; i < N; i++){
    for (int j = 0; j < NR; j++){
      for (int k = 0; k < NC; k++){
        out(i,j,k) = first(i,j,k) ^ other(i,j,k);
      }
    }
  }
  return out;
}

template<int x_size, int h_n>
class gru {
  float alpha_z, alpha_r, alpha_h;
  binary_tensor<x_size, h_n, h_n+1> B_z, B_r, B_h;
  binary_tensor<x_size, h_n, 1> h, h_final, rt, zt, ht; //ht = h_twiddle
  binary_tensor<x_size, h_n+1, 1> hx, rhx;

public:
  void setup(){
    ; //random tensors for B's, random floats for alpha's
  }

  binary_tensor<x_size, h_n, 1> forward(binary_tensor<x_size,1,1> input){

//    std::cout << sizeof(binary_tensor<x_size, h_n, 1>) << " " << sizeof(Eigen::Tensor<bool,3>(x_size, h_n, 1)) << std::endl;

//    //first inputs
//    hx = concatenate(h, input);
//    zt = (B_z,hx);
//    rt = (B_r,hx);
//    rhx = concatenate(rt * h, input);
//    //now make the update for h
//    ht = (B_h,rhx) * zt;
//    ~zt;
//    h *= zt; //this guy has sigm(1-alpha_z)
//    h_final = h + ht;
//    alpha_h = _helper::sigm(1.f-alpha_z) + tanhf(alpha_h * _helper::sigm(alpha_r))*_helper::sigm(alpha_z); //update all the float stuff
//    return h_final;
  }

};

#endif //DLIB_RNN_BINARY_HPP
