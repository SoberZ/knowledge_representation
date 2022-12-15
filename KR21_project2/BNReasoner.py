import os
# OpenMP problem fix. Remove if it causes problems.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from typing import Union
from BayesNet import BayesNet
import pandas as pd
import networkx as nx
from functools import reduce
import itertools


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
        return self

    @staticmethod
    def marginalization(factor: pd.DataFrame, variable: str):
        """
        :param factor: The factor dataframe for which cpt you want to marginalize
        :param variable: The variable which you want to marginalize (sum-out)
        """
        if variable in factor:
            group_cols = factor.columns.tolist()
            group_cols.remove(variable)
            group_cols.remove('p')
            df2 = factor.groupby(group_cols, as_index=False)['p'].sum()
            return df2
        return False


    def maxing_out(self, factor: pd.DataFrame, variable: str):
        """
        A new Dataframe is made per CPT in which the rows are sorted on score and the duplicates
        with the lowest score are removed. The self.bn is then updated using the new dataframe.

        :param variable: The variable which you want to max-out on
        """

        # A new Dataframe is made per CPT in which the rows are sorted on score and the duplicates
        # with the lowest score are removed. The self.bn is then updated using the new dataframe.
       # for current_var in self.bn.get_all_variables():
        #    if variable in self.bn.get_cpt(current_var).columns:
         #       new_list = []
          #      newer_list = []
           #     df_new = self.bn.get_cpt(
            #        current_var).sort_values('p', ascending=False)
             #   df_new = df_new.drop_duplicates(
              #      subset=self.bn.get_cpt(current_var).columns.difference(
             #           [variable, 'p'])).sort_index()
             #   for item in df_new.loc[:, variable].tolist():
             #       new_list.append(variable + '= ' + str(item))
             #   if 'extended_factors' in df_new.columns:
             #       for i, item in enumerate(df_new['extended_factors']):
             #           newer_list.append(item + ',' + new_list[i])
             #       df_new['extended_factors'] = newer_list

             #   else:
             #       df_new['extended_factors'] = new_list
             #   # df_new['extended_factors'].append(new_extended_factor)
             #   df_new = df_new.drop(labels=variable, axis='columns')
             #   self.bn.update_cpt(current_var, df_new)
        #return self

        # There is only one column to maximize over.
        if len(factor.columns) == 2 and factor.columns[0] == variable:
            return factor

        df_new = factor.sort_values('p', ascending=False)
        df_new = df_new.drop_duplicates(subset=factor.columns.difference([variable, 'p'])).sort_index()
        new_extended_factor = variable + '=' + str(df_new.loc[:, variable].tolist()[0])
        if 'extended_factors' in df_new.columns:
            df_new['extended_factors'] = \
                df_new['extended_factors'] + ',' + new_extended_factor
        else:
            df_new['extended_factors'] = new_extended_factor
        df_new = df_new.drop(labels=variable, axis='columns')
        return df_new


    def find_path_DFS(self, reasoner, start, end):
        """
        Find a path between two nodes using DFS.
        If there exists a path, this means we have
        no d-separation so we return False.
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

    def find_path_BFS(self, reasoner, start, end):
        """
        Find a path between two nodes using BFS.
        If there exists a path, this means we have
        no d-separation so we return False.
        If there is no such a path, return True.
        """
        visited = [] # The nodes that have been visited
        queue = [start] # Nodes to find paths from.
        while len(queue) > 0:
                node = queue.pop()
                if node == end:
                    return False

                neighbors = list(nx.all_neighbors(reasoner.bn.structure, node))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
                        visited.append(neighbor)

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

        self.bn.draw_structure()

        res = True
        for start in X:
            for end in Y:
                # End in evidence is d-separated.
                if end in Z:
                    res = res and True
                else:
                    res = res and self.find_path_DFS(reasoner_copy, start, end)
        return res


    def d_separation_BFS(self, X, Y, Z):
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
                    res = res and self.find_path_BFS(reasoner_copy, start, end)
        return res

    def independence(self, X, Y, Z):
        """
        Given three sets of variables X, Y, and Z,
        determine whether X is independent of Y given Z.
        D-separation implies independence.
        """
        # Parents to the evidence since
        # non_Descendants are conditionally independent
        for x in X:
            Z += (self.bn.structure.predecessors(x))

        if self.d_separation(X, Y, Z):
            return True

        return False

    def factor_multiplication(self, fac_f: pd.DataFrame, fac_g: pd.DataFrame) -> pd.DataFrame:
        """
        Given two factors f and g, compute the multiplied factor h=fg
        :param fac_f, fac_g: The factors you want to multiply.
        """
        common_vars = [i for i in fac_f.columns if i in fac_g.columns]

        if common_vars[0] == "p":
            f_len = len(fac_f.columns) - 1
            g_len = len(fac_g.columns) - 1
            n_vars = f_len + g_len
            worlds = [list(i) for i in itertools.product([False, True], repeat=n_vars)]
            new_ps = []
            for world in worlds:
                part1, part2 = world[0:f_len], world[f_len:]
                p1_series = pd.Series(dict(zip(fac_f.columns, part1)))
                p2_series = pd.Series(dict(zip(fac_g.columns, part2)))
                cit1 = self.bn.get_compatible_instantiations_table(p1_series, fac_f)
                cit2 = self.bn.get_compatible_instantiations_table(p2_series, fac_g)
                new_ps.append(list(cit1.p)[0] * list(cit2.p)[0])

            transposed_worlds = [*zip(*worlds)]
            new_df = {}
            for i, column in enumerate(list(fac_f.columns[:-1]) + list(fac_g.columns[:-1])):
                new_df[column] = transposed_worlds[i]
            new_df["p"] = new_ps
            return pd.DataFrame(new_df)

        merged = pd.merge(fac_f, fac_g, on=common_vars[0], how='outer')
        for i in range(len(merged.index)):
            merged.at[i, 'p'] = merged.at[i, 'p_x'] * merged.at[i, 'p_y']

        merged.drop(['p_x', 'p_y'], axis=1, inplace=True)

        return merged



    def variable_elimination(self, variables: list) -> pd.DataFrame:
        """
        Sum out a set of variables by using variable elimination.
        """
        # Get all cpts
        cpt_dict = self.bn.get_all_cpts()
        partial_factors = None

        for variable in variables:
            # Step 1. Join all factors containing that variable
            cpts = []
            if partial_factors is not None:
                cpts.append(partial_factors)

            partial_factors = None
            new_dict = {}
            for cpt in cpt_dict:
                if variable in cpt_dict[cpt].columns:
                    cpts.append(cpt_dict[cpt])
                else:
                    new_dict[cpt] = cpt_dict[cpt]
            cpt_dict = new_dict
            joined_factors = reduce(lambda x, y: self.factor_multiplication(x, y), cpts)

            # Step 2. Sum out the influence of the variable on new factor
            partial_factors = self.marginalization(joined_factors, variable)

        return partial_factors


    def maximum_a_posteriori(self, evidence: list):
        """
        Compute maximum a-posteriory instantiation + value of query variables
        Q given a possibly empty evidence e.
        """
        # Remove non-evidence from the joint distribution
        q_not_e = [i for i in self.bn.get_all_variables() if i not in evidence]
        marginal_distribution = self.variable_elimination(q_not_e)

        for e in evidence:
            marginal_distribution = self.maxing_out(marginal_distribution, e)

        return marginal_distribution


    def maximum_a_posteriori_marginalize(self, evidence: list):
        """
        Compute maximum a-posteriory instantiation + value of query variables
        Q given a possibly empty evidence e.
        """
        # Remove non-evidence from the joint distribution
        q_not_e = [i for i in self.bn.get_all_variables() if i not in evidence]

        # Join all tables containing the q/e
        all_cpts = self.bn.get_all_cpts().items()
        res = list(filter(lambda x: any(q in x[1].columns for q in q_not_e), all_cpts))
        res = [a for _, a in res]
        marginal_distribution = reduce(lambda x, y: self.factor_multiplication(x, y), res)

        # Then sum out q_not_e
        for e in q_not_e:
            marginal_distribution = self.marginalization(marginal_distribution, e)

        # Max out evidence
        for e in evidence:
            marginal_distribution = self.maxing_out(marginal_distribution, e)

        return marginal_distribution

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
            p = cpt.loc[:, 'p'].tolist()
            for item in p:
                if item > highest:
                    highest = item
                    highest_cpt = cpt
        return self.cpt_to_dict(highest_cpt, evidence_dict, highest), 'p='+str(highest)

    @staticmethod
    def cpt_to_dict(cpt, evidence_dict, highest):
        """
        :param cpt:
        :return:
        """
        dict_cpt = dict()
        variables = cpt.loc[:, 'extended_factors'][[cpt.index[cpt['p'] == highest][0]]].tolist()
        for column in cpt.columns:
            if column not in ('p', 'extended_factors'):
                dict_cpt[column] = cpt.loc[:, column].tolist()[0]
        for item in variables:
            item = item.split(',')
            for it in item:
                var, value = it.split('= ')
                dict_cpt[var] = value
        for keys in dict_cpt:
            dict_cpt[keys] = str(dict_cpt[keys])
        for item,item_value in evidence_dict.items():
            dict_cpt[item] = item_value
        return dict_cpt

    def new_edge_counter(self, int_graph):
        """
        Given an interaction graph and a variable, the number of new edges when removing the variable
        is calculated
        :param int_graph: The interaction graph in which the variable is present
        """

        new_edges_dict = {}
        for variable in int_graph.nodes():
            number_of_new_edges = 0
            for nb1 in list(int_graph.neighbors(variable)):
                for nb2 in list(int_graph.neighbors(variable)):
                    if nb1 != nb2 and not int_graph.has_edge(nb1, nb2):
                        number_of_new_edges += 1
            new_edges_dict[variable] = number_of_new_edges

        return new_edges_dict

    def ordering(self, heuristic):
        """
        Given a set of variables X in the Bayesian network, compute a good ordering for
        the elimination of X based on the min-degree heuristics and the min-fill heuristics
        :param heuristic: The heuristic you want to use for the ordering.
        """
        order = list()

        if heuristic == "min-degree":
            interaction_graph = self.bn.get_interaction_graph()
            while interaction_graph.number_of_nodes() > 0:
                min_edges = min(dict(interaction_graph.degree()
                                     ).items(), key=lambda x: x[1])
                order.append(min_edges[0])
                for variable1 in list(interaction_graph.neighbors(min_edges[0])):
                    for variable2 in list(interaction_graph.neighbors(min_edges[0])):
                        if variable1 != variable2 and not interaction_graph.has_edge(variable1, variable2):
                            interaction_graph.add_edge(variable1, variable2)
                interaction_graph.remove_node(min_edges[0])

        elif heuristic == "min-fill":
            interaction_graph = self.bn.get_interaction_graph()
            while interaction_graph.number_of_nodes() > 0:
                min_edges = min(self.new_edge_counter(interaction_graph),
                                key=self.new_edge_counter(interaction_graph).get)
                order.append(min_edges)
                for variable1 in list(interaction_graph.neighbors(min_edges)):
                    for variable2 in list(interaction_graph.neighbors(min_edges)):
                        if variable1 != variable2 and not interaction_graph.has_edge(variable1, variable2):
                            interaction_graph.add_edge(variable1, variable2)
                interaction_graph.remove_node(min_edges)

        else:
            print("ERROR: please choose between min-degree or min-fill.")

        return order[:len(self.bn.get_all_variables())]



if __name__ == "__main__":
    # Hardcoded voorbeeld om stuk te testen
    # BN1 = BNReasoner('testing/test.BIFXML')
    # BN2 = BNReasoner('testing/lecture_example.BIFXML')
    # BN3 = BNReasoner('testing/lecture_example2.BIFXML')
    BN4 = BNReasoner('testing/dog_problem.BIFXML')


    # var_elim = BN4.variable_elimination(["bowel-problem", "family-out"])
    # print(var_elim)

    max_b = BN4.maximum_a_posteriori(["bowel-problem", "hear-bark"])
    print(max_b)
    max_a = BN4.maximum_a_posteriori_marginalize(["bowel-problem", "hear-bark"])
    print(max_a)
    # check = BN.independence(["Slippery Road?"], ["Sprinkler?"], ["Winter?", "Rain?"])
    # for variable in BN2.bn.get_all_variables():
    #     print(BN2.bn.get_cpt(variable))
    # print('\n')
    # print(BN2.most_probable_explanation({'Rain?': True, 'Winter?': False}))
    # print('\n\n')
    # BN.network_pruning('Rain?', False)
    # BN.bn.draw_structure()
    # for variable in BN3.bn.get_all_variables():
    #     print(BN3.bn.get_cpt(variable))
    # BN2.bn.draw_structure()

