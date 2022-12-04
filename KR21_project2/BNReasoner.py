from typing import Union
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go
    def network_pruning(self, variable, evidence):
        """
        :param variable: Variable which has known evidence to prune network with
        :param evidence: Assignment of the variable: True/False
        """
        # Remove rows from cpt which do not agree with the evidence of the variable
        for current_var in self.bn.get_all_variables():
            CPT = self.bn.get_cpt(current_var)
            if variable in CPT:
                CPT = CPT[CPT[variable] == evidence]
            self.bn.update_cpt(current_var, CPT)

        # Remove all edges which have the selected variable as first node (directed)
        remove_set = set()
        for node_1, node_2 in self.bn.structure.edges():
            if node_1 == str(variable):
                remove_set.add((node_1, node_2))
        for node_tuple in remove_set:
            self.bn.del_edge((node_tuple[0], node_tuple[1]))

    def marginalization(self):
        pass

    def maxing_out(self, variable):
        """
        :param variable: The variable which you want to max-out on
        """
        # A new Dataframe is made per CPT in which the rows are sorted on score and the duplicates
        # with the lowest score are removed. The self.bn is then updated using the new dataframe.
        for current_var in self.bn.get_all_variables():
            if variable in self.bn.get_cpt(current_var):
                df_new = self.bn.get_cpt(current_var)\
                    .sort_values('p', ascending=False).drop_duplicates(
                    subset=self.bn.get_cpt(current_var).columns.difference(
                        [variable, 'p'])).sort_index()
                df_new = df_new.drop(labels=variable ,axis='columns')
                self.bn.update_cpt(current_var, df_new)


if __name__ == "__main__":
    BN = BNReasoner('testing/lecture_example.BIFXML')
    BN.network_pruning('Rain?', False)