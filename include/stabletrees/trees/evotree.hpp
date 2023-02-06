#pragma once
#include "tree.hpp"
#include "probabalistictree.hpp"

using namespace std;



class EvoTree{

    public:
        EvoTree();
        EvoTree(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int random_state);
        dVector predict(dMatrix  &X);
        void learn(dMatrix &X, dVector &y);
        void update(dMatrix &X, dVector &y, int size, int generations);
        Node* get_root();
        Tree* tree; 
        tuple<ProbabalisticTree*, ProbabalisticTree*> breed(dMatrix &X, dVector &y, ProbabalisticTree* tree1, ProbabalisticTree* tree2);
        std::vector<ProbabalisticTree*> selection(std::vector<ProbabalisticTree*> population, int N);
        void fitness_function(std::vector<ProbabalisticTree*> &population,dMatrix &X, dVector &y, dVector &yprev);
        std::vector<ProbabalisticTree*> create_population(dMatrix &X, dVector &y, int N);
        std::vector<ProbabalisticTree*> generate_population(dMatrix &X, dVector &y, std::vector<ProbabalisticTree*> population, int size, double childprop, double eliteprop);
        

    protected:
        int random_state;
        int _criterion;
        int max_depth;
        double min_split_sample;
        int min_samples_leaf;
        bool adaptive_complexity;
        double mse(dVector &ypred, dVector &y);
};



EvoTree::EvoTree(){
    tree = new Tree();
    int max_depth = INT_MAX;
    double min_split_sample = 2.0;
    _criterion = 0;
    adaptive_complexity = false;
    this->min_samples_leaf = 1;
    random_state =0;
}
EvoTree::EvoTree(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int random_state){
    tree = new Tree(_criterion, max_depth,  min_split_sample, min_samples_leaf, adaptive_complexity);
    this->random_state = random_state;
    if(max_depth !=NULL){
       this-> max_depth =  max_depth;
    }else{
        this-> max_depth =  INT_MAX;
    }
    this-> min_split_sample = min_split_sample;
    this->min_samples_leaf = min_samples_leaf;
    this->_criterion = _criterion;
    this->adaptive_complexity = adaptive_complexity;
}

void EvoTree::learn(dMatrix &X, dVector &y){
    tree->learn(X,y);
   
}

dVector EvoTree::predict(dMatrix &X){
    return tree->predict(X);
}

Node* EvoTree::get_root(){
    return tree->root;
}

double EvoTree::mse(dVector &ypred, dVector &y){
    return (y.array() - ypred.array() ).square().mean();
}

void EvoTree::fitness_function(std::vector<ProbabalisticTree*> &population,dMatrix &X, dVector &y, dVector &yprev){
    dMatrix scores = dMatrix::Zero(population.size(), 3);
    for (size_t i = 0; i < population.size(); i++)
    {   
        dVector ypred = population[i]->predict(X);
        scores(i,0) = mse(y, ypred);
        scores(i,1) = mse(yprev, ypred);
        scores(i,2) = population[i]->tree_depth;
    }
    dVector maxes = scores.colwise().sum();

    for (size_t i = 0; i < population.size(); i++)
    {   
        population[i]->fitness = 0.5* scores(i,0)/maxes(0,1) + scores(i,1)/maxes(1,1) + 0.5* scores(i,2)/maxes(2,1);
    }
    
    std::sort(population.begin( ), population.end( ), [](const ProbabalisticTree* a, const ProbabalisticTree* b) { return a->fitness < b->fitness; });
    

}


tuple<ProbabalisticTree*, ProbabalisticTree*> EvoTree::breed(dMatrix &X, dVector &y, ProbabalisticTree* tree1, ProbabalisticTree* tree2){
    //printf("breed \n");
    tree1 = tree1->copy();
    tree2 = tree2->copy();
    std::vector<Node*> node_list1 = tree1->make_node_list();
    std::vector<Node*> node_list2 = tree2->make_node_list();
    //std::vector<int> possible_nodes1;
    //std::vector<int> possible_nodes2;
    std::vector<int> possible_nodes;
    int n = min(node_list1.size(), node_list2.size());
    for(int i = 0; i<n; i++){
        
        if( node_list1[i] != NULL && !node_list1[i]->is_leaf() && node_list2[i] != NULL && !node_list2[i]->is_leaf() ){
            possible_nodes.push_back(i);
        }
    }

    // for (size_t i = 0; i < possible_nodes.size(); i++)
    //     {   
    //         printf("possible_nodes %d \n", possible_nodes[i]);
    //     }
    // for(int i = 0; i<node_list1.size(); i++){
        
    //     if(node_list1[i] != NULL && !node_list1[i]->is_leaf()){
    //         possible_nodes1.push_back(i);
    //     }
    // }
    // for(int i = 0; i<node_list2.size(); i++){
    //     if( node_list2[i] != NULL && !node_list2[i]->is_leaf()){
    //         possible_nodes2.push_back(i);
    //     }
    // }
    //printf("possible_nodes1 %d \n" , possible_nodes.size());
    
    std::mt19937 gen(this->random_state);
    this->random_state = 36969*(this->random_state & 0177777) + (this->random_state>>16) + 1;
    std::uniform_int_distribution<int>  distr(0, possible_nodes.size()-1);

    int crossover_index1 = possible_nodes[distr(gen)];
    int crossover_index2 = possible_nodes[distr(gen)];


    // std::mt19937 gen2(this->random_state);
    // std::uniform_int_distribution<int>  d(0, possible_nodes2.size());
    // gen.seed(this->random_state);
    // int crossover_index2 = possible_nodes2[d(gen2)];

    //printf("crossover_index1 %d crossover_index2 %d \n", crossover_index1,crossover_index2);

    Node* node1 = node_list1[crossover_index1]->copy();

    Node* node2 = node_list2[crossover_index2]->copy();
    //printf("before crossover %d\n", node2->get_split_feature());
    tree1->root = tree1->crossover(X,y,node2, crossover_index1);
    //printf("first crossover\n");
    tree2->root  = tree2->crossover(X,y,node1, crossover_index2);
    //printf("second crossover\n");
    

    return tuple<ProbabalisticTree*, ProbabalisticTree*>(tree1,tree2);

}



std::vector<ProbabalisticTree*> EvoTree::create_population(dMatrix &X, dVector &y, int N){
    std::vector<ProbabalisticTree*> population(N);
    for (size_t i = 0; i < N; i++)
    {   
        std::mt19937 gen(this->random_state);
        this->random_state = 36969*(this->random_state & 0177777) + (this->random_state>>16) + 1;
        std::uniform_int_distribution<int>  distr(2, 11);
        int depth = distr(gen);
        std::mt19937 gen2(this->random_state);
        std::uniform_int_distribution<int>  distr2(5, 11);
        double min_sample_split = (double)distr2(gen2);
        ProbabalisticTree* tree = new ProbabalisticTree(0,depth, min_sample_split,1,false, this->random_state + i);
        tree->learn(X,y);
        //printf("%d \n", tree->root->get_split_feature());
        population[i]= (tree);
    }
    return population;
}

std::vector<ProbabalisticTree*> EvoTree::generate_population(dMatrix &X, dVector &y, std::vector<ProbabalisticTree*> population, int size, double childprop, double eliteprop){
    int nchilds = int((size*childprop)/2);
    int nelites = int(size*eliteprop);
    int nfreshblood = size - 2*nchilds - nelites;
    std::vector<ProbabalisticTree*> new_population;
    for (int i = 0; i < nchilds; i++)
    {
        std::vector<ProbabalisticTree*> keep = selection(population,2);
        ProbabalisticTree* parent1 = keep[0];
        ProbabalisticTree* parent2 = keep[1];
        ProbabalisticTree* child1;
        ProbabalisticTree* child2;
        tie(child1,child2) = breed(X,y,parent1,parent2);
        //printf("%d success! \n", i);
        new_population.push_back(child1);
        new_population.push_back(child2);
    }
    for (int i = 0; i < nelites; i++)
    {
        new_population.push_back(population[i]);
    }
    std::vector<ProbabalisticTree*> freshblood = create_population(X,y,nfreshblood);
    for (size_t i = 0; i < nfreshblood; i++)
    {
        new_population.push_back(freshblood[i]);
    }

    return new_population;
    
}


std::vector<ProbabalisticTree*> EvoTree::selection(std::vector<ProbabalisticTree*> population, int N){
    //https://en.wikipedia.org/wiki/Stochastic_universal_sampling
    double F = 0; //total fitness of Population
    for (size_t i = 0; i < population.size(); i++)
    {
        F+= population[i]->fitness;
    }
    double P = F/N; //distance between the pointers (F/N)

    //random number between 0 and P
    std::mt19937 gen(this->random_state);
    this->random_state = 36969*(this->random_state & 0177777) + (this->random_state>>16) + 1;
    std::uniform_int_distribution<int>  distr(0, P);
    int start = distr(gen);
    
    std::vector<double> pointers(N);
    for (int i = 0; i < N; i++)
    {
        pointers[i] = start + i*P;
    }
    std::vector<ProbabalisticTree*> keep;
    for (auto p : pointers){
        int i = 0;
        while (true)
        {
            double s = 0;
            for (size_t j = 0; j < i+1; j++)
            {
                s+= population[j]->fitness;
            }
            if( s>=p){
                break;
            }
            i+=1;
        }
        keep.push_back(population[i]);
    }
    
    return keep;
}





void EvoTree::update(dMatrix &X, dVector &y, int size, int generations){
    dVector yprev = predict(X);
    std::vector<ProbabalisticTree*> population = create_population(X,y,size);
    for (size_t i = 0; i < generations; i++)
    {
        //printf("%d \n", i);
        fitness_function(population, X,y,yprev);
        // for (size_t i = 0; i < population.size()-1; i++)
        // {   
        //     printf("%d, %f, %f\n", population[i]->fitness <= population[i+1] ->fitness, population[i]->fitness,population[i+1]->fitness);
        // }
        if( (i+1)==1 || (i+1)%10 ==0 || (i+1) ==generations+1){
            printf("generation %d: 1. %f 2. %f 3. %f \n" , i+1, population[0]->fitness, population[1]->fitness,population[2]->fitness  );
        }

        population = generate_population(X,y,population,size, 0.5,0.25);

    }
    fitness_function(population, X,y,yprev);
    this->tree = population[0]->copy();
} 




