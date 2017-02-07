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

using net_type = dlib::binary_conv<32,2,2,1,1, dlib::input<dlib::matrix<unsigned char>>>;

int main(int argc, char *argv[]){
  net_type net;

  dlib::matrix<unsigned char, 2, 2> sample;
  sample = 2,2,1,1;

  net(sample);


  return 0;
}