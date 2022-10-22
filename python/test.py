import os
cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.append(cur_file_path + '\\..\\cpp\\build\\Release\\')

from stable_trees import Node


# == End import CPP wraper .so file == #

import numpy as np
import unittest

class TestKF(unittest.TestCase):
    def test_can_construct_with_x_and_v(self):
        split_value =1.0
        prediction = 1.0
        n_samples = 1
        split_feature =1
        split_score = 1
        node= Node(split_value,split_score,split_feature,n_samples, prediction)

        self.assertTrue(node.is_leaf())
        left= Node(split_value,split_score,split_feature,n_samples, prediction)
        right= Node(split_value,split_score,split_feature,n_samples, prediction)
        node.set_left_node(left)
        node.set_right_node(right)
        self.assertFalse(node.is_leaf())
        print(node.get_left_node(),node.predict(),node.get_left_node().predict())
        self.assertEqual(node.predict(),prediction )


    


if __name__ == '__main__':
    unittest.main()