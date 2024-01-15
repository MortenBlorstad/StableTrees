

cimport cython
import numpy as np
cimport numpy as cnp
from cloudpickle import CloudPickler
cnp.import_array()

ctypedef cnp.npy_float32 DTYPE_t          # Type of X
ctypedef cnp.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef cnp.npy_intp SIZE_t              # Type for indices and counters
ctypedef cnp.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef cnp.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef class Node:
    # Declare the data types of the attributes
    cdef public float split_value
    cdef public int split_feature
    cdef public int num_samples
    cdef public float prediction
    cdef public float y_var
    cdef public float w_var
    cdef public float score
    cdef public SIZE_t[::1] features_indices
    cdef public Node left_child
    cdef public Node right_child
    cdef public float parent_expected_max_S
    
    # Constructor
    def __init__(self, float split_value, float score, int split_feature, 
                 int num_samples, float prediction, float y_var, 
                 float w_var, SIZE_t[::1] features_indices):
        self.split_value = split_value
        self.split_feature = split_feature
        self.num_samples = num_samples
        self.prediction = prediction
        self.y_var = y_var
        self.w_var = w_var
        self.score = score
        self.features_indices = features_indices
        

    cpdef bint is_leaf(self):
            return self.left_child is None and self.right_child is None 

    cpdef int get_split_feature(self):
        return self.split_feature

    cpdef float get_impurity(self):
        return self.score

    cpdef SIZE_t[::1] get_features_indices(self):
        return self.features_indices

    cpdef float get_split_value(self):
        return self.split_value

    cpdef float predict(self):
        return self.prediction

    cpdef int nsamples(self):
        return self.num_samples  


     # Add a custom __reduce__ method
    def __reduce__(self):
        # Serialize the object using CloudPickler
        pickler = CloudPickler()
        state = self.__getstate__()

        # Return a tuple with the constructor and state
        return (self.__class__, (), state)

    # Add a __getstate__ method to return the object's state
    cdef dict __getstate__(self):
        state = {
            'split_value': self.split_value,
            'split_feature': self.split_feature,
            'num_samples': self.num_samples,
            'prediction': self.prediction,
            'y_var': self.y_var,
            'w_var': self.w_var,
            'score': self.score,
            'features_indices': list(self.features_indices),
        }
        return state  