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
            print(self.bn.get_cpt('Rain?'))
        else:
            self.bn = net

    # TODO: This is where your methods should go
    def network_pruning(self, variable='Rain?', evidence='False'):
        """
        :param variable: Variable which has known evidence to prune network with
        :param evidence: Assignment of the variable: True/False
        """
        print(variable,evidence)


        for index, row in self.bn.get_cpt(variable).iterrows():
            print(self.bn.get_all_cpts())
            if str(row[variable]) != str(evidence):
                self.bn.update_cpt(variable, self.bn.get_cpt(variable).drop(index-1))
        print(self.bn.get_all_cpts())
        self.bn.all
        self.bn.del_edge((variable,node2))
        self.bn.draw_structure()
        print(self.bn.get_cpt(variable))
        pass

    def marginalization(self):
        pass

    def maxing_out(self):
        pass


if __name__ == "__main__":
    BN = BNReasoner('testing/lecture_example.BIFXML')
    BN.network_pruning('Rain?', 'False')