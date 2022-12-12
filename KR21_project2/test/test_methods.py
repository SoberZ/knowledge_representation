import unittest
from KR21_project2 import BNReasoner as BNR


def cpts_to_answer():

    pass

class MyTestCase(unittest.TestCase):
    def test_network_pruning(self):
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        pruned_BN = BN.network_pruning('Rain?', False)
        self.assertEqual(pruned_BN.get_all_cpts(), False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
