//
// Created by Aman LaChapelle on 2/11/17.
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


#ifndef DLIB_RNN_NTM_HPP
#define DLIB_RNN_NTM_HPP

#include <map>
#include <vector>
#include <iostream>

#include <dlib/dnn.h>
#include <dlib/dnn/tensor.h>
#include <dlib/dnn/tensor_tools.h>

using namespace dlib;

template<
        typename SUBNET,
        int input_dim, int output_dim, int mem_size = 128,
        int mem_dim = 20, int controller_dim=100,
        int controller_layer_size=1, int shift_range=1,
        int write_head_size=1, int read_head_size=1
        >
class ntm_ { //implement first in barebones python or just straight up linear algebra c++ with none of the DNN stuff? Math isn't bad tbh

  struct head { //apply all the nonlinearities
    dlib::resizable_tensor w_hid_to_k, b_hid_to_k;
    dlib::resizable_tensor w_hid_to_gate, b_hid_to_gate;
    int num_shifts;
    dlib::resizable_tensor w_hid_to_shift, b_hid_to_shift;
    dlib::resizable_tensor w_hid_to_gamma, b_hid_to_gamma;
  };

  struct write_head : public head {
    dlib::resizable_tensor w_hid_to_erase, b_hid_to_erase;
    dlib::resizable_tensor w_hid_to_add, b_hid_to_add;

    dlib::resizable_tensor write(dlib::resizable_tensor htm1, dlib::resizable_tensor wtm1, dlib::resizable_tensor Mt) {
      dlib::resizable_tensor et, at, Mtp1, identity(identity_matrix<float>(mat(Mt))), wtm1e, at_w;

      //erase op
      tt::gemm(0.0, et, 1.0, w_hid_to_erase, false, htm1, false);
      tt::add(1.0, et, 1.0, b_hid_to_erase);
      tt::relu(et, et);

      //add op
      tt::gemm(0.0, at, 1.0, w_hid_to_add, false, htm1, false);
      tt::add(1.0, at, 1.0, b_hid_to_add);
      tt::relu(at, at);

      //update memory
      tt::gemm(0.0, wtm1e, 1.0, wtm1, false, et, false);
      cuda::scale_tensor(wtm1e, -1.f);
      tt::add(1.0, identity, 1.0, wtm1e); //1-w*e

      tt::gemm(0.0, Mtp1, 1.0, Mt, false, identity, false); //Mt-1(1-w*e)

      tt::gemm(0.0, at_w, 1.0, wtm1, false, at, false); //w*a

      tt::add(1.0, Mtp1, 1.0, at_w); //M = M~ + w*a

      return Mtp1;

    }
  };

  struct read_head : public head {
    dlib::resizable_tensor read(dlib::resizable_tensor wtm1, dlib::resizable_tensor Mt){
      tt::gemm(0.0, wtm1, 1.0, Mt, false, wtm1, false);

      return wtm1;
    }
  };

  //change this to be better
  template<typename SUB> using controller_net_t = relu<fc<controller_dim, SUB>>;

  controller_net_t<SUBNET> controller;
  std::vector<dlib::resizable_tensor> M, read_w, read, write_w, output, hidden;

  dlib::resizable_tensor k, s;
  float beta, g, gamma;

  dlib::tt::tensor_rand rand;
  int idx;

  //need similarities

  dlib::matrix<float> cosine_similarity(dlib::matrix<float> &one, dlib::matrix<float> &two) {
    DLIB_CASSERT(one.nc() == two.nr()); //objects are compatible
    DLIB_CASSERT(two.nc() == 1); //two is a vector only

    //get norms
    float one_norm = sqrt(sum(trans(one)*one));
    float two_norm = sqrt(sum(trans(two)*two));

    dlib::matrix<float> out = one * two;
    out /= one_norm*two_norm + 1e-3;

    return out;
  }


//  dlib::resizable_tensor calc_wc() {
//    dlib::resizable_tensor wc;
//
//    wc = cosine_similarity(mat(M[idx]), mat(k));
//    cuda::scale_tensor(wc, beta);
//    tt::softmax(wc, wc);
//
////    std::cout << wc.num_samples() << " " << wc.k() << " " << wc.nr() << " " << wc.nc() << std::endl;
//
//    return wc;
//  }

//  dlib::resizable_tensor calc_wg(dlib::resizable_tensor wc) {
//    dlib::resizable_tensor wg;
//    tt::add(0.0, wg, g, wc);
//    tt::add(1.0, wg, (1.f-g), read_w[idx]);
//  }

public:
  ntm_(){
    k.set_size(1,1,mem_size,1);
    cuda::set_tensor(k, 1.f);
    M.push_back(resizable_tensor(1,1,mem_dim,mem_size));
  }

  void init_state() {

    idx = 0;

    //initialize memory
    M.push_back(dlib::resizable_tensor(1, 1, mem_size, mem_dim));

    //initialize read weights and read list
    dlib::resizable_tensor rw (1,1,mem_size,mem_size);
    dlib::resizable_tensor r (1,1,mem_dim,mem_dim);
    for (int i = 0; i < read_head_size; i++){
      rand.fill_gaussian(rw, 0.0, 0.1);
      dlib::tt::softmax(rw, rw);
      read_w.push_back(rw);

      rand.fill_gaussian(r, 0.0, 0.1);
      dlib::tt::tanh(r, r);
      read.push_back(r);
    }

    //initialize write weights
    dlib::resizable_tensor ww (1,1,mem_size,mem_size);
    for (int i = 0; i < write_head_size; i++){
      rand.fill_gaussian(ww, 0.0, 0.1);
      dlib::tt::softmax(ww, ww);
      write_w.push_back(ww);
    }

    //initialize output and hidden controller states
    dlib::resizable_tensor oh (1,1,controller_dim,controller_dim);
    for (int i = 0; i < controller_layer_size; i++){
      rand.fill_gaussian(oh, 0.0, 0.1);
      dlib::tt::tanh(oh, oh);
      output.push_back(oh);
      hidden.push_back(oh);
    }

    idx++;

  }



};

#endif //DLIB_RNN_NTM_HPP
