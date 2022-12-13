import unittest
import sys
import pandas as pd
sys.path.insert(1, "../")
import BNReasoner as BNR
import BayesNet


def cpts_to_answer():

    pass


class MyTestCase(unittest.TestCase):
    def test_network_pruning(self):
        # Setup answer in dataframe to test the method with
        answer_Data = {'Winter?': [False,True],'Rain?':[False,False], 'p': [0.9,0.2]}
        test_df = pd.DataFrame(data=answer_Data)

        # Create BN from example and perform network_pruning on it
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        pruned_BN = BN.network_pruning('Rain?', False).bn.get_cpt('Rain?')

        # Check if pruned dataframe is the same as the correct answer
        for i, j in zip(test_df.iterrows(), pruned_BN.iterrows()):
            self.assertEqual(str(i[1].to_string()), str(j[1].to_string()))

    def test_marginalization(self):
        pass


if __name__ == '__main__':
    unittest.main()
