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

#include "include/binary.hpp"

int main(int argc, char *argv[]){

  gru<3,3> gru1;

  std::cout << binary_tensor<2,4,4>().size() << " " << Eigen::TensorFixedSize<bool, Eigen::Sizes<2,4,4>>().size() << std::endl;

  binary_tensor<1,2,2> me;
  me(0,0,0) = true;
  me(0,0,1) = false;
  me(0,1,0) = false;
  me(0,1,1) = true;
//  for (int i = 0; i < 2; i++){
//    for (int j = 0; j < 2; j++){
//      for (int k = 0; k < 2; k++){
//        me(i,j,k) = true;
//      }
//    }
//  }
  binary_tensor<1,2,1> mult;
  mult(0,0,0) = true;
  mult(0,1,0) = false;
//  mult(1,0,0) = true;
//  mult(1,1,0) = false;

  auto out = (me,mult);

//  auto out = gru1.forward(mult);
  std::cout << out(0,0,0) << " " << out(0,1,0) << std::endl;
  std::cout << out.n() << " " << out.nr() << " " << out.nc() << std::endl;

  return 0;
}