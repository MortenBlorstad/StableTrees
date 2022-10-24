#pragma once

#ifndef __NODE_HPP_INCLUDED__

#define __NODE_HPP_INCLUDED__

#include <stdio.h>
#include <iostream>
using namespace std;

class Node{
    public:
        Node* left_child;
        Node* right_child;
        double split_value= NULL;
        double prediction= NULL;
        int n_samples= NULL;
        int split_feature= NULL;
        double split_score = NULL;

        
        Node(double _split_value,double _split_score, int _split_feature, int _n_samples, double _prediction);
        Node(double _prediction, int _n_samples);

        Node* get_left_node();
        Node* get_right_node();
        void set_left_node(Node* node);
        void set_right_node(Node* node);
        bool is_leaf();
        double predict();
        string text();
        int nsamples();
        double get_split_score();
    
    };

Node::Node(double _split_value, double _split_score, int _split_feature,int _n_samples, double _prediction){
    split_value = _split_value;
    split_score = _split_score;
    n_samples = _n_samples;
    split_feature = _split_feature;
    left_child=NULL;
    right_child=NULL;
    prediction = _prediction;

}

Node::Node(double _prediction, int _n_samples){
    n_samples = _n_samples;
    prediction = _prediction;
    left_child=NULL;
    right_child=NULL;
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
    return this->left_child;
}
Node* Node::get_right_node(){

    return this->right_child ;
}

string Node::text(){
    char text [50];
    if(!is_leaf()){
        
        sprintf (text, "X_%d <= %f", this->split_feature, this->split_value);

    }else{
        sprintf (text, "%f",this->prediction);
    }

    return text;
}

int Node::nsamples(){
    return this->n_samples;
}

double Node::get_split_score(){
    return this-> split_score;
}

bool Node::is_leaf(){
    return this->left_child== NULL && this->right_child== NULL;
    
}


#endif