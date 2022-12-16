import unittest
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
        marginalized_BN = BN.marginalization(BN.bn.get_cpt('Rain?'), 'Rain?')
        # Check if pruned dataframe is the same as the correct answer
        for i, j in zip(test_df.iterrows(), marginalized_BN.iterrows()):
            self.assertEqual(str(i[1].to_string()), str(j[1].to_string()))

    def test_maxing_out(self):
        # Setup answer in dataframe to test the method with
        answer_Data = {'Winter?': [False, True], 'p': [0.9, 0.8], 'extended_factors': ['Rain?= False','Rain?= True']}
        test_df = pd.DataFrame(data=answer_Data)

        # Create BN from example and perform network_pruning on it
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        maxed_out_BN = BN.maxing_out(BN.bn.get_cpt('Rain?'), 'Rain?')
        # Check if pruned dataframe is the same as the correct answer
        for i, j in zip(test_df.iterrows(), maxed_out_BN.iterrows()):
            self.assertEqual(str(i[1].to_string()), str(j[1].to_string()))

    def test_d_separation(self):
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        response1 = BN.d_separation(['Rain?'], ['Sprinkler?'], ['Winter?'])  # Not d-separated
        response2 = BN.d_separation(['Rain?'], ['Sprinkler?'], ['Winter?', 'Wet Grass?'])  # d-separated
        self.assertEqual(response1, True)
        self.assertEqual(response2, False)

    def test_independence(self):
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        response1 = BN.d_separation(['Rain?'], ['Sprinkler?'], ['Winter?'])  # Not d-separated
        response2 = BN.d_separation(['Rain?'], ['Sprinkler?'], ['Winter?', 'Wet Grass?'])  # d-separated
        self.assertEqual(response1, True)
        self.assertEqual(response2, False)

    def test_factor_multiplication(self):
        # Initialising example and running method
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        df1 = BN.bn.get_cpt('Rain?')
        df2 = BN.bn.get_cpt('Slippery Road?')
        outcome = BN.factor_multiplication(df1,df2)
        # Expected DataFrame
        answer_Data = {'Winter?': [False, False, True, True, False, False, True, True], 'Rain?':
            [False, False, False, False, True, True, True, True], 'Slippery Road?':
            [False, True, False, True, False, True, False, True], 'p':
            [0.90, 0.00, 0.20, 0.00, 0.03, 0.07, 0.24, 0.56]}
        expected = pd.DataFrame(data=answer_Data)

        for i, j in zip(expected.iterrows(), outcome.iterrows()):
            self.assertEqual(str(i[1].to_string()), str(j[1].to_string()))

    def test_variable_elimination(self):
        # initiate dataframe and use method for its outcome
        BN = BNR.BNReasoner('testing/dog_problem.BIFXML')
        outcome = BN.variable_elimination(["family-out", "dog-out", "light-on"])
        # setup correct answer in dataframe
        answer_Data = {'bowel-problem': [False, False, True, True],'hear-bark':
            [False, True, False, True], 'p': [1.42, 2.58, 1.42, 2.58]}
        expected = pd.DataFrame(data=answer_Data)
        # check for each row if the outcome equals the expected outcome
        for i, j in zip(expected.iterrows(), outcome.iterrows()):
            self.assertEqual(str(i[1].to_string()), str(j[1].to_string()))

    def test_most_probable_explanation(self):
        # Load lecture example
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        # Run method with a test_evidence dictionary
        test_evidence = {'Rain?': True, 'Winter?': False}
        test_outcome = (BN.most_probable_explanation(test_evidence)[0])
        # Test the outcome of the method against the correct outcome
        correct_answer = {'Rain?': True, 'Wet Grass?': 'True', 'Sprinkler?': 'True', 'Winter?': False}
        self.assertEqual(test_outcome,correct_answer)
        # Test the probability
        correct_p_value = 'p=0.95'
        self.assertEqual(correct_p_value, BN.most_probable_explanation(test_evidence)[1])

    def test_ordering(self):
        # Load lecture example
        BN = BNR.BNReasoner('testing/lecture_example.BIFXML')
        # Run method for both heuristics
        test_case_1 = BN.ordering('min-degree')
        test_case_2 = BN.ordering('min-fill')
        # Expected orders
        expected_result_1 = ['Slippery Road?', 'Winter?', 'Sprinkler?', 'Rain?', 'Wet Grass?']
        expected_result_2 = ['Winter?', 'Sprinkler?', 'Wet Grass?', 'Rain?', 'Slippery Road?']
        # Test if the methods are correct
        self.assertEqual(test_case_1, expected_result_1)
        self.assertEqual(test_case_2, expected_result_2)

    def test_marginal_distributions(self):
        BN1 = BNR.BNReasoner('testing/test.BIFXML')
        outcome = BN1.marginal_distribution(['C'], 'A', True)
        # setup correct answer in dictionary
        expected = {'C False': 0.6, 'C True': 0.4}
        # check for each row if the outcome equals the expected outcome
        self.assertEqual(outcome, expected)

    def test_MAP(self):
        BN4 = BNR.BNReasoner('testing/dog_problem.BIFXML')
        outcome_b = BN4.maximum_a_posteriori(["bowel-problem", "hear-bark"])
        outcome_a = BN4.maximum_a_posteriori_marginalize(["bowel-problem", "hear-bark"])
        answer_Data_a = {'p': [2.4381], 'extended_factors':
            ['bowel-problem= False,hear-bark= True']}
        answer_Data_b = {'p': [2.58], 'extended_factors':
            ['bowel-problem= False,hear-bark= True']}
        expected_a = pd.DataFrame(data=answer_Data_a)
        expected_b = pd.DataFrame(data=answer_Data_b)
        # check for each row if the outcome equals the expected outcome
        for i, j in zip(expected_a.iterrows(), outcome_a.iterrows()):
            self.assertEqual(str(i[1].to_string()), str(j[1].to_string()))
        for i, j in zip(expected_b.iterrows(), outcome_b.iterrows()):
            self.assertEqual(str(i[1].to_string()), str(j[1].to_string()))


if __name__ == '__main__':
    unittest.main()
