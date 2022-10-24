#include "treebuilder.hpp"


#include "node.hpp"
#include "splitter.hpp"
#include <stdio.h>
#include <iostream>
using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;


using namespace std;
using namespace Eigen;

int main(){

    Treebuilder* tb = new Treebuilder();
    tb->example();
    
    std::cout << tb->root->is_leaf()<< std::endl;
}