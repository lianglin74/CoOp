#!nosetests --nocapture
import unittest


class TestCloudStorage(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_leaf_nodes(self):
        from qd.cloud_storage import get_leaf_names
        all_fname = ['a/b', 'a/b/c']
        leaf_names = get_leaf_names(all_fname)

        self.assertEqual(len(leaf_names), 1)
        self.assertEqual(leaf_names[0], 'a/b/c')

    def test_get_leaf_nodes2(self):
        from qd.cloud_storage import get_leaf_names
        all_fname = ['a/b', 'a/b/c', 'a/b/d']
        leaf_names = get_leaf_names(all_fname)

        self.assertEqual(len(leaf_names), 2)
        self.assertIn('a/b/c', leaf_names)
        self.assertIn('a/b/d', leaf_names)
