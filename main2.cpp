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
#include <dlib/data_io.h>

using namespace dlib;
using net_type = fc<10,rnn<GRU, BIDIRECTIONAL, 1, 5, 10, input<matrix<float>>>>;

int main(int argc, char *argv[]){

//  std::vector<matrix<unsigned char>> training_images;
//  std::vector<unsigned long> training_labels;
//  std::vector<matrix<unsigned char>> testing_images;
//  std::vector<unsigned long> testing_labels;
//  load_mnist_dataset("/Users/Aman/Desktop/data/", training_images, training_labels, testing_images, testing_labels);

  net_type net;
//  dnn_trainer<net_type> trainer(net);
//  trainer.set_learning_rate(0.01);
//  trainer.set_min_learning_rate(0.00001);
//  trainer.be_verbose();

//  trainer.set_synchronization_file("test_sync", std::chrono::seconds(20));
//
//  std::vector<matrix<unsigned char>> samples;
//  std::vector<unsigned long> labels;
//  dlib::rand r(time(0));
//  while (trainer.get_learning_rate() >= 1e-3){
//    samples.clear();
//    labels.clear();
//
//    while (samples.size() < 32){
//      int idx = r.get_random_32bit_number() % training_images.size();
//      samples.push_back(training_images[idx]);
//      labels.push_back(training_labels[idx]);
//    }
//
//    trainer.train_one_step(samples, labels);
//
//  }
//
//  trainer.get_net();

  std::vector<dlib::matrix<float,5,2>> samples;
  dlib::matrix<float,5,2> sample;
  sample = 1,2,3,4,5,6,7,8,9,1;
  samples.push_back(sample);


  std::cout << mat(net(sample));
//  std::cout << labels[0];

//  std::cout << net(output,3)


  return 0;
}