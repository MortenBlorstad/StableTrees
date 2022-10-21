#pragma once

#ifndef __NODE_HPP_INCLUDED__

#define __NODE_HPP_INCLUDED__
#include <stdio.h>
class Node{
    public:
        Node* left_child;
        Node* right_child;
        Node();
        
        Node* get_left_node();
        Node* get_right_node();
        bool is_leaf();
    };


Node::Node(){
    this->left_child = NULL;
    this->right_child = NULL;
}

Node* Node::get_left_node(){
    return this->left_child ;
}
Node* Node::get_right_node(){
    return this->right_child ;
}


bool Node::is_leaf(){
    return  this->left_child == NULL && this->right_child == NULL;
}


#endif