#pragma once
#include "tree.hpp"
#include "node.hpp"
#include "probabalisticsplitter.hpp"


class ProbabalisticTree: public Tree{
    public:
        ProbabalisticTree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int random_state);
        ProbabalisticTree();
        
        virtual void learn(dMatrix  &X, dVector &y);
        Node* crossover_rec(dMatrix  &X, dVector &y, Node* node1, Node* node2 ,int index, int swap_index, int depth);
        Node* crossover(dMatrix  &X, dVector &y, Node* node2 , int swap_index);
        virtual ProbabalisticTree* copy();
        double fitness; 
    private:
        int random_state;
        
};

 ProbabalisticTree* ProbabalisticTree::copy(){
    ProbabalisticTree* tree = new ProbabalisticTree(*this);
    tree->root = root->copy();
    return tree;
 }

ProbabalisticTree::ProbabalisticTree():Tree(){
    Tree();
    random_state = 0;
}

ProbabalisticTree::ProbabalisticTree(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int random_state):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity){
    Tree(_criterion, max_depth,  min_split_sample, min_samples_leaf, adaptive_complexity);
    this->random_state = random_state;
}
Node* ProbabalisticTree::crossover(dMatrix  &X, dVector &y, Node* node2 ,int swap_index){
    Node* swap_node = node2;
    Node* node = crossover_rec(X,y, root, swap_node,0, swap_index, 0 );
    return node;
}

Node* ProbabalisticTree::crossover_rec(dMatrix  &X, dVector &y, Node* node1, Node* node2 ,int index, int swap_index, int depth){
    if(y.size()<1){
        return NULL;
    }
    if(node1->is_leaf()){
        return node1;
    }
    int left_index = index*2+1;
    int right_index = index*2+2;
    //printf(" %d %d %d\n", node1->get_split_feature(),left_index,right_index );
    dVector feature = X.col(node1->split_feature);
    iVector mask_left;
    iVector mask_right;
    tie(mask_left, mask_right) = get_masks(feature, node1->split_value);
    if(mask_left.size()<1 || mask_right.size()<1 ){
        //printf("left mask %d, right mask%d  \n",mask_left.size(), mask_right.size());
        node1->left_child = NULL;
        node1->right_child = NULL;
    }else{
        iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
        dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
        dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
        //printf(" %d %d %d\n",swap_index, left_index, right_index );
        if (left_index == swap_index){
            //printf("L %d %d %d\n",swap_index, left_index, node2->get_split_feature() );
            //Node* node = update_tree_info(X_left, y_left, node2);
            //printf("L node %d \n", node->get_split_feature());
            node1->left_child = update_tree_info(X_left,y_left,node2,depth+1); //node ;
            //printf("L after %d \n", node1->left_child->get_split_feature());

        }else if(right_index == swap_index){
            //printf("R %d %d %d\n",swap_index, right_index, node2->get_split_feature() );
            
            node1->right_child = update_tree_info(X_right,y_right, node2,depth+1); //update_tree_info(X_right, y_right, node2) ;
        }else{
            //printf("C %d %d %d\n",swap_index, left_index, right_index );
            node1->left_child = crossover_rec(X_left,y_left, node1->left_child, node2, left_index, swap_index, depth+1);
            node1->right_child = crossover_rec(X_right,y_right, node1->right_child, node2, right_index, swap_index, depth+1);
        }
        
    } 
    //printf("end %d \n", node1->left_child->get_split_feature());
    return node1;
}


void ProbabalisticTree::learn(dMatrix  &X, dVector &y){
    if(adaptive_complexity){
        cir_sim = cir_sim_mat(100, 100);
        grid_end = 1.5*cir_sim.maxCoeff();
    }
    total_obs = y.size();
    splitter = new ProbabalisticSplitter(min_samples_leaf,total_obs,_criterion, adaptive_complexity, this->random_state);
    if(adaptive_complexity){
        splitter->cir_sim = cir_sim;
        splitter->grid_end = grid_end;
    }
    root = build_tree(X, y, 0);
}





