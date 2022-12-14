import unittest
import sys
import pandas as pd
import BNReasoner as BNR



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
        # Setup answer in dataframe to test the method with
        answer_Data = {'Winter?': [False, True], 'p': [1.0, 1.0]}
        test_df = pd.DataFrame(data=answer_Data)

        # Create BN from example and perform network_pruning on it
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        print(BN.bn.get_cpt('Rain?'),'\n\n\n')
        marginalized_BN = BN.marginalization(BN.bn.get_cpt('Rain?'), 'Rain?')

        print(marginalized_BN)
        # Check if pruned dataframe is the same as the correct answer
        for i, j in zip(test_df.iterrows(), marginalized_BN.iterrows()):
            self.assertEqual(str(i[1].to_string()), str(j[1].to_string()))

    def test_maxing_out(self):
        # Setup answer in dataframe to test the method with
        answer_Data = {'Winter?': [False, True], 'p': [0.9, 0.8], 'extended_factors': ['Rain?= False','Rain?= False']}
        test_df = pd.DataFrame(data=answer_Data)

        # Create BN from example and perform network_pruning on it
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        maxed_out_BN = BN.maxing_out('Rain?').bn.get_cpt('Rain?')

        # Check if pruned dataframe is the same as the correct answer
        for i, j in zip(test_df.iterrows(), maxed_out_BN.iterrows()):
            self.assertEqual(str(i[1].to_string()), str(j[1].to_string()))

    def test_d_separation(self):
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        response1 = BN.d_separation(['Rain?'], ['Sprinkler?'], ['Winter?'])  # Not d-separated
        response2 = BN.d_separation(['Rain?'], ['Sprinkler?'], ['Winter?', 'Wet Grass?'])  # d-separated
        self.assertEqual(response1, False)
        self.assertEqual(response2, True)

    def test_independence(self):
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        response1 = BN.d_separation(['Rain?'], ['Sprinkler?'], ['Winter?'])  # Not d-separated
        response2 = BN.d_separation(['Rain?'], ['Sprinkler?'], ['Winter?', 'Wet Grass?'])  # d-separated
        self.assertEqual(response1, False)
        self.assertEqual(response2, True)

    def test_factor_multiplication(self):
        pass

    def test_variable_elimination(self):
        pass

    def test_most_probable_explanation(self):
        # Load lecture example
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        test_evidence = {'Rain?': True, 'Winter?': False}
        # Determine correct output to compare with
        correct_answer = ({'Rain?': 'True', 'Sprinkler?': 'True', 'Wet Grass?': 'True'}, 'p=0.95')
        # Run comparison
        self.assertEqual(BN.most_probable_explanation(test_evidence), correct_answer)

    def test_ordering(self):
        pass

    def test_marginal_distributions(self):
        pass

    def test_MAP(self):
        pass


if __name__ == '__main__':
    unittest.main()
