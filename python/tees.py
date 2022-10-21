import os
cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.append(cur_file_path + '\\..\\cpp\\build\\Release\\')

from stable_trees import Node


n = Node()
print(n.get_right_node())
print(n.is_leaf())
