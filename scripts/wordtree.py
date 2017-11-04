import os
from anytree import Node, RenderTree
from anytree.dotexport import RenderTreeGraph
from anytree import PreOrderIter


class WordTree(object):
    def __init__(self, file_path):
        self._roots = []
        self._tree_file_path = os.path.dirname(file_path)
        self._map = {}  # map for fast access
        # Read the tree
        self.read_tree(file_path)

    def read_tree(self, file_path):
        """Read a word tree
        """
        _nodes = []
        with open(file_path) as tree_f:
            for line_idx, line in enumerate(tree_f):
                name, parent_idx = line.split()
                parent_idx = int(parent_idx)
                if parent_idx < 0:
                    node = Node(name, used=0)
                    self._roots.append(node)
                else:
                    node = Node(name, parent=_nodes[parent_idx], used=0)
                _nodes.append(node)
                self._map[node.name] = node

    def __repr__(self):
        return '\n'.join(["{}{}{}".format(pre, node.name,
                                          ':{}'.format(node.used) if node.used else '')
                          for root in self._roots
                          for pre, _, node in RenderTree(root)
                          ])

    def __str__(self):
        return ' '.join(['{}:{}'.format(root.name, len(root.descendants)) for root in self._roots])

    def __getitem__(self, key):
        return self._map[key]

    def to_dot(self):
        for root in self._roots:
            RenderTreeGraph(root).to_dotfile(os.path.join(self._tree_file_path, root.name) + '.dot')

    def __iter__(self):
        for root in self._roots:
            for node in PreOrderIter(root):
                yield node
