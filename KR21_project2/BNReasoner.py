import os
# OpenMP problem fix. Remove if it causes problems.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from typing import Union
from BayesNet import BayesNet
import networkx as nx


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

    def marginalization(self, variable):
        """
        :param variable: The variable which you want to marginalize (sum-out)
        """
        for current_var in self.bn.get_all_variables():
            if variable in self.bn.get_cpt(current_var):
                group_cols = self.bn.get_cpt(current_var).columns.tolist()
                group_cols.remove(variable)
                group_cols.remove('p')
                df2 = self.bn.get_cpt(current_var).groupby(group_cols, as_index=False)['p'].sum()
                self.bn.update_cpt(current_var, df2)

    def maxing_out(self, variable):
        """
        :param variable: The variable which you want to max-out on
        """
        # A new Dataframe is made per CPT in which the rows are sorted on score and the duplicates
        # with the lowest score are removed. The self.bn is then updated using the new dataframe.
        for current_var in self.bn.get_all_variables():
            if variable in self.bn.get_cpt(current_var):
                df_new = self.bn.get_cpt(current_var).sort_values('p', ascending=False)
                df_new = df_new.drop_duplicates(subset=self.bn.get_cpt(current_var).columns.difference([variable, 'p'])).sort_index()
                df_new['extended_factors'] = df_new.loc[:, variable]
                df_new = df_new.drop(labels=variable ,axis='columns')
                self.bn.update_cpt(current_var, df_new)
                print(self.bn.get_cpt(current_var))

    def find_path(self, reasoner, start, end):
        """
        Find a path between two nodes. If there exists a path,
        this means we have no d-separation so we return False.
        If there is no such a path, return True.
        """
        visited = [] # The nodes that have been visited
        search_nodes = [start] # Nodes to find paths from.
        while len(search_nodes) > 0:
                node = search_nodes.pop()

                if node == end:
                    return False

                if node not in visited:
                    visited.append(node)
                    neighbors = list(nx.all_neighbors(reasoner.bn.structure, node))
                    for neighbor in neighbors:
                        search_nodes.append(neighbor)
        return True

    def d_separation(self, X, Y, Z):
        """
        X, Y - Sets of Nodes to check for d-separation
        Z - Evidence
        D-Separation algorithm. X and Y are d-separated by Z iff every
        path between a node in X to a node in Y is d-blocked by Z.
        """
        reasoner_copy = BNReasoner(self.bn)
        for evidence in Z:
            reasoner_copy.network_pruning(evidence, True)

        res = True
        for start in X:
            for end in Y:
                # End in evidence is d-separated.
                if end in Z:
                    res = res and True
                else:
                    res = res and self.find_path(reasoner_copy, start, end)
        return res

    def independence(self):
        pass

    def variable_elimination(self):
        pass

if __name__ == "__main__":
    # Hardcoded voorbeeld om stuk te testen
    BN = BNReasoner('testing/lecture_example.BIFXML')
    chekc = BN.d_separation(["Slippery Road?"], ["Wet Grass?", "Winter?"], ["Winter?"])
    print(chekc)
    # BN.bn.draw_structure()
