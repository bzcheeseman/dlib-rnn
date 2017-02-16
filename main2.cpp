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

#include <iostream>

//#include "include/ntm.hpp"
//#include "include/binary.hpp"
#include "include/cudnn_rnn.hpp"
#include <dlib/dnn.h>

using namespace dlib;
using net_type = rnn<GRU, BIDIRECTIONAL, 1, 10, 5, input<matrix<float>>>;

int main(int argc, char *argv[]){
  net_type net;

  dlib::matrix<float, 5, 1> sample;
  sample = 2,1,1,2,3;

  dlib::matrix<float> output = mat(net(sample));
  std::cout << output << std::endl;

  std::cout << mat(net(output)); //this is how you would run predictions for sequences or something!

//  ntm_<dlib::con<32,3,3,1,1,dlib::input<dlib::matrix<unsigned char>>>, 1,1> ntm;


  return 0;
}