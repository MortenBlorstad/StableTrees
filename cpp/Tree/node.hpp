#pragma once

#ifndef __NODE_HPP_INCLUDED__

#define __NODE_HPP_INCLUDED__

#include <stdio.h>

class Node{
    public:
        Node* left_child;
        Node* right_child;
        double split_value;
        double prediction;
        int n_samples;
        int split_feature;
        double split_score;

        
        Node(double _split_value,double _split_score, int _split_feature, int _n_samples, double _prediction);
        Node(double _prediction, int _n_samples);

        Node* get_left_node();
        Node* get_right_node();
        void set_left_node(Node* node);
        void set_right_node(Node* node);
        bool is_leaf();
        double predict();
    
    };

Node::Node(double _split_value, double _split_score, int _split_feature,int _n_samples, double _prediction){
    split_value = _split_value;
    split_score = _split_score;
    n_samples = _n_samples;
    split_feature = _split_feature;
    
    prediction = _prediction;
    this->left_child = NULL;
    this->right_child = NULL;
}

Node::Node(double _prediction, int _n_samples){
    n_samples = _n_samples;
    prediction = _prediction;
    this->left_child = NULL;
    this->right_child = NULL;
}

void Node::set_left_node(Node* node){
    this->left_child = node;
}
void Node::set_right_node(Node* node){
    this->right_child = node;
}

double Node::predict(){
    return this->prediction;
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