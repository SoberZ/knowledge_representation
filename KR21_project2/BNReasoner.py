import os
from typing import Union
from BayesNet import BayesNet
import networkx as nx

# OpenMP problem fix. Remove if it causes problems.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
            cpt = self.bn.get_cpt(current_var)
            if variable in cpt:
                cpt = cpt[cpt[variable] == evidence]
            self.bn.update_cpt(current_var, cpt)

        # Remove all edges which have the selected variable as first node (directed)
        remove_set = set()
        for node_1, node_2 in self.bn.structure.edges():
            if node_1 == str(variable):
                remove_set.add((node_1, node_2))
        for node_tuple in remove_set:
            self.bn.del_edge((node_tuple[0], node_tuple[1]))

    def marginalization(self, factor, variable):
        """
        :param factor: The factor for which cpt you want to marginalize
        :param variable: The variable which you want to marginalize (sum-out)
        """
        if variable in self.bn.get_cpt(factor):
            group_cols = self.bn.get_cpt(factor).columns.tolist()
            group_cols.remove(variable)
            group_cols.remove('p')
            df2 = self.bn.get_cpt(factor).groupby(group_cols, as_index=False)['p'].sum()
            self.bn.update_cpt(factor, df2)

    def maxing_out(self, variable):
        """
        :param variable: The variable which you want to max-out on
        """
        # A new Dataframe is made per CPT in which the rows are sorted on score and the duplicates
        # with the lowest score are removed. The self.bn is then updated using the new dataframe.
        for current_var in self.bn.get_all_variables():
            if variable in self.bn.get_cpt(current_var).columns:
                df_new = self.bn.get_cpt(current_var).sort_values('p', ascending=False)
                df_new = df_new.drop_duplicates(
                    subset=self.bn.get_cpt(current_var).columns.difference(
                        [variable, 'p'])).sort_index()
                new_extended_factor = variable + '= ' + str(df_new.loc[:, variable].tolist()[0])
                if 'extended_factors' in df_new.columns:
                    df_new['extended_factors'] = \
                        df_new['extended_factors'] + ',' + new_extended_factor
                else:
                    df_new['extended_factors'] = new_extended_factor
                # df_new['extended_factors'].append(new_extended_factor)
                df_new = df_new.drop(labels=variable, axis='columns')
                self.bn.update_cpt(current_var, df_new)
                print(self.bn.get_cpt(current_var))
                print('\n')

    def find_path(self, reasoner, start, end):
        """
        Find a path between two nodes. If there exists a path,
        this means we have no d-separation so we return False.
        If there is no such a path, return True.
        """
        visited = []  # The nodes that have been visited
        search_nodes = [start]  # Nodes to find paths from.
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

    def independence(self, X, Y, Z):
        """
        Given three sets of variables X, Y, and Z,
        determine whether X is independent of Y given Z.
        D-separation implies independence.
        """
        return self.d_separation(X, Y, Z)

    def variable_elimination(self):
        """

        """
        pass

    def most_probable_explanation(self, evidence_dict):
        """
        :param evidence_dict:
        """
        evidence_set = evidence_dict.keys()
        variable_set = set(self.bn.get_all_variables())
        mpe_set = variable_set.difference(evidence_set)
        for variable, evidence in evidence_dict.items():
            self.network_pruning(variable, evidence)
        # maximize-out all variables in mpe_set
        for variable in mpe_set:
            self.maxing_out(variable)
        highest = 0
        highest_cpt = None
        for variable in self.bn.get_all_variables():
            cpt = self.bn.get_cpt(variable)
            p = cpt.loc[:, 'p'].tolist()[0]

            if p > highest:
                highest = p
                highest_cpt = cpt
        return self.cpt_to_dict(highest_cpt), 'p='+str(highest)

    @staticmethod
    def cpt_to_dict(cpt):
        """
        :param cpt: 
        :return: 
        """
        dict_cpt = dict()
        p_value = cpt.loc[:, 'p'].tolist()[0]
        variables = cpt.loc[:, 'extended_factors'].tolist()[0]
        variables = variables.split(',')
        for column in cpt.columns:
            if column not in ('p', 'extended_factors'):
                dict_cpt[column] = cpt.loc[:, column].tolist()[0]
        for pair in variables:
            pair = pair.split('= ')
            var, value = pair
            dict_cpt[var] = value
        return dict_cpt


if __name__ == "__main__":
    # Hardcoded voorbeeld om stuk te testen
    BN = BNReasoner('testing/lecture_example.BIFXML')
    # check = BN.independence(["Slippery Road?"], ["Sprinkler?"], ["Winter?", "Rain?"])
    # print(check)
    for variable in BN.bn.get_all_variables():
        print(BN.bn.get_cpt(variable))
    print('\n\n')
    BN.network_pruning('Rain?', False)
    # BN.bn.draw_structure()
    for variable in BN.bn.get_all_variables():
        print(BN.bn.get_cpt(variable))
    # print('highest=', BN.most_probable_explanation({'Rain?': True, 'Winter?': False}))
