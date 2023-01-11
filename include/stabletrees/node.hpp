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
        double split_value= NULL; // split value
        double prediction= NULL; // node's prediction 
        int n_samples= NULL; // number of sample in node
        int split_feature= NULL; //  index of feature to split
        double split_score = NULL; // split reduduction
        double impurity; // node impurity


        
        Node(double _split_value,double _impurity, double _split_score, int _split_feature, int _n_samples, double _prediction);
        Node(double _prediction, int _n_samples);

        Node* get_left_node();
        Node* get_right_node();
        void set_left_node(Node* node);
        void set_right_node(Node* node);
        bool is_leaf();
        double predict();
        int nsamples();
        int get_split_feature();
        double get_split_value();
        double get_split_score();
        double get_impurity();
    
    };

Node::Node(double _split_value, double _impurity, double _split_score, int _split_feature,int _n_samples, double _prediction){
    split_value = _split_value;
    impurity = _impurity;
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
int Node::get_split_feature(){
    return this->split_feature;
}


double Node::get_split_value(){
    return this->split_value;
}

Node* Node::get_left_node(){
    return this->left_child;
}
Node* Node::get_right_node(){

    return this->right_child ;
}

int Node::nsamples(){
    return this->n_samples;
}

double Node::get_split_score(){
    return this-> split_score;
}

double Node::get_impurity()
{
    return this-> impurity;
}

bool Node::is_leaf(){
    return this->left_child== NULL && this->right_child== NULL;
    
}


#endif