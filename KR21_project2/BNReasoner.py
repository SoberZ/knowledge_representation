import os
# OpenMP problem fix. Remove if it causes problems.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from typing import Union
from BayesNet import BayesNet
import pandas as pd
pd.set_option('display.max_colwidth', None)
import networkx as nx
from functools import reduce
import itertools
import copy
from collections import deque


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
        return factor

    def maxing_out(self, factor: pd.DataFrame, variable: str):
        """
        A new Dataframe is made per CPT in which the rows are sorted on score and the duplicates
        with the lowest score are removed. The self.bn is then updated using the new dataframe.

        :param variable: The variable which you want to max-out on
        """

        # A new Dataframe is made per CPT in which the rows are sorted on score and the duplicates
        # with the lowest score are removed. The self.bn is then updated using the new dataframe.

        # There is only one column to maximize over.
        if len(factor.columns) == 2 and factor.columns[0] == variable:
            return factor.loc[factor['p'] == factor['p'].max()]

        new_list = []
        newer_list = []
        df_new = factor.sort_values('p', ascending=False)
        df_new = df_new.drop_duplicates(
            subset=factor.columns.difference(
                [variable, 'p'])).sort_index()
        if variable in df_new.columns:
            for item in df_new.loc[:, variable].tolist():
                new_list.append(variable + '= ' + str(item))
            if 'extended_factors' in df_new.columns:
                for i, item in enumerate(df_new['extended_factors']):
                    newer_list.append(item + ',' + new_list[i])
                df_new['extended_factors'] = newer_list
            else:
                df_new['extended_factors'] = new_list
            df_new = df_new.drop(labels=variable, axis='columns')
        return df_new


    def d_separation(self, start, end, evidence):
        """
        X, Y - Sets of Nodes to check for d-separation
        Z - Evidence
        D-Separation algorithm. X and Y are d-separated by Z iff every
        path between a node in X to a node in Y is d-blocked by Z.
        """
        # Phase 1: insert all ancestors of Z in X
        visit_nodes = copy.copy(evidence)
        obs = set()

        for node in visit_nodes:
            for parent in self.bn.structure.predecessors(visit_nodes.pop()):
                obs.add(parent)

        # Phase 2
        visited = []  # The nodes that have been visited
        search_nodes = [(s, True) for s in start]  # Nodes to find paths from.

        while len(search_nodes) > 0:
            node, up = search_nodes.pop()

            if (node, up) not in visited:
                visited.append((node, up))

                if node not in evidence and node in end:
                    return False

                if up and node not in evidence:
                    for parent in self.bn.structure.predecessors(node):
                        search_nodes.append((parent, True))
                    for child in self.bn.structure.successors(node):
                        search_nodes.append((child, False))
                elif not up:
                    if node not in evidence:
                        for child in self.bn.structure.successors(node):
                            search_nodes.append((child, False))

                    if node in evidence or node in obs:
                        for parent in self.bn.structure.predecessors(node):
                            search_nodes.append((parent, True))
        return True


    # def d_separation_pruning(self, start, end, evidence):
    #     combined = start + end + evidence
    #     reasoner_copy = BNReasoner(self.bn)
    #     G = reasoner_copy.bn.structure

    #     # Remove leaf nodes
    #     leaf_nodes = deque([n for n in G.nodes if G.out_degree[n] == 0])
    #     while leaf_nodes:
    #         leaf = leaf_nodes.popleft()
    #         if leaf not in combined:
    #             for pred in G.predecessors(leaf):
    #                 if G.out_degree[pred] == 1:
    #                     leaf_nodes.append(pred)
    #             G.remove_node(leaf)

    #     # Remove outgoing evidence edges
    #     for e in evidence:
    #         reasoner_copy.network_pruning(e, True)



    #     reasoner_copy.bn.draw_structure()

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

    def multiply_factors(self, f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
        f_len, g_len = len(f.columns) - 1, len(g.columns) - 1
        n_vars = f_len + g_len
        worlds = [list(i) for i in itertools.product([False, True], repeat=n_vars)]
        new_ps = []

        for world in worlds:
            part1, part2 = world[0:f_len], world[f_len:]
            p1_series = pd.Series(dict(zip(f.columns, part1)))
            p2_series = pd.Series(dict(zip(g.columns, part2)))
            cit1 = self.bn.get_compatible_instantiations_table(p1_series, f)
            cit2 = self.bn.get_compatible_instantiations_table(p2_series, g)
            new_ps.append(list(cit1.p)[0] * list(cit2.p)[0])

        # Create new table
        transposed_worlds = [*zip(*worlds)]
        new_df = {}
        for i, column in enumerate(list(f.columns[:-1]) + list(g.columns[:-1])):
            new_df[column] = transposed_worlds[i]
        new_df["p"] = new_ps
        res = pd.DataFrame(new_df)
        return res

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
                # print(partial_factors)
                cpts.append(partial_factors)
            partial_factors = None

            # Check if variable is in the cpt
            new_dict = {}
            for cpt in cpt_dict:
                if variable in cpt_dict[cpt].columns:
                    cpts.append(cpt_dict[cpt])
                else:
                    # Rest of the cpts
                    new_dict[cpt] = cpt_dict[cpt]
            cpt_dict = new_dict

            joined_factors = reduce(lambda x, y: self.multiply_factors(x, y), cpts)

            # Step 2. Sum out the influence of the variable on new factor
            partial_factors = self.marginalization(joined_factors, variable)

        return partial_factors

    def variable_elimination2(self, factor, variables: list) -> pd.DataFrame:
        """
        Sum out a set of variables by using variable elimination.
        """
        # Get all cpts
        cpt_dict = factor
        partial_factors = None

        for variable in variables:
            # Step 1. Join all factors containing that variable
            cpts = []
            if partial_factors is not None:
                cpts.append(partial_factors)
            partial_factors = None

            # Check if variable is in the cpt
            new_dict = {}
            for cpt in cpt_dict:
                if variable in cpt_dict[cpt].columns:
                    cpts.append(cpt_dict[cpt])
                else:
                    # Rest of the cpts
                    new_dict[cpt] = cpt_dict[cpt]
            cpt_dict = new_dict

            joined_factors = reduce(lambda x, y: self.multiply_factors(x, y), cpts)

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

        return self.maxing_out(marginal_distribution, 'p')


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
        marginal_distribution = reduce(lambda x, y: self.multiply_factors(x, y), res)

        # Then sum out q_not_e
        for e in q_not_e:
            marginal_distribution = self.marginalization(marginal_distribution, e)

        # Max out evidence
        for e in evidence:
            marginal_distribution = self.maxing_out(marginal_distribution, e)

        return self.maxing_out(marginal_distribution, 'p')


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
        highest = 0
        highest_cpt = None
        for variable in mpe_set:
            for all_vars in self.bn.get_all_variables():
                if variable in self.bn.get_cpt(all_vars).columns:
                    maxed_out = self.maxing_out(self.bn.get_cpt(all_vars), variable)
                    self.bn.update_cpt(variable, maxed_out)
                    cpt = self.bn.get_cpt(variable)
                    p = cpt.loc[:, 'p'].tolist()
                    for item in p:
                        if item >= highest:
                            highest = item
                            highest_cpt = cpt
        return self.cpt_to_dict(highest_cpt, evidence_dict, highest), 'p='+str(highest)

    def mpe(self, evidence: list):
        """
        Most probable explanation
        """
        # Remove all evidence from the joint distribution
        q_not_e = [i for i in self.bn.get_all_variables() if i not in evidence]
        marginal_distribution = self.variable_elimination(evidence)

        for e in q_not_e:
            marginal_distribution = self.maxing_out(marginal_distribution, e)

        return self.maxing_out(marginal_distribution, 'p')


    @staticmethod
    def cpt_to_dict(cpt, evidence_dict, highest):
        """
        :param cpt:
        :return:
        """
        dict_cpt = dict()
        variables = cpt.loc[:, 'extended_factors'][[cpt.index[cpt['p'] == highest][0]]].tolist()
        for item in variables:
            item = item.split(',')
            for it in item:
                var, value = it.split('= ')
                dict_cpt[var] = value
        for keys in dict_cpt:
            dict_cpt[keys] = str(dict_cpt[keys])
        for item, item_value in evidence_dict.items():
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


    def get_path(self, start, end):
        """
        :param start: variable to start the path from
        :param end: variable to end the path
        :return: a list of the path from the start to end variable
        """
        network = self.bn.structure

        if nx.has_path(network, source=start, target=end):
            return nx.shortest_path(network, source=start, target=end)


    def marginal_distribution(self, query_variables: list, variable, evidence):
        """
        :param query_variables: The query variable against which you want to test the evidence
        :param variable: The variable for which you have evidence
        :param evidence: Assignment of the variable: True/False
        """
        if not variable:
            to_remove = [var for var in self.bn.get_all_variables() if var not in query_variables]
            return self.variable_elimination(to_remove)

        # Reduce all factors wrt e
        res = reduce(lambda x, y: self.multiply_factors(x, y), list(self.bn.get_all_cpts().values()))
        res = res[res[variable] == evidence] # all factors wrt e

        # This is Pr(Q^e)
        not_Q_e = [v for v in self.bn.get_all_variables() if v != variable and v not in query_variables]
        pr_Q_e = self.variable_elimination(not_Q_e) # all columns not e or Q
        pr_Q_e = pr_Q_e[pr_Q_e[variable] == evidence]

        # Compute Pr(Q|e) = Pr(Q^e)/Pr(e)
        not_e = [v for v in self.bn.get_all_variables() if v != variable]
        pr_e = self.variable_elimination(not_e) # all columns not e
        pr_e_p = pr_e[pr_e[variable] == evidence]['p']

        res_dict = {}
        for item in query_variables:
            for i, row in pr_Q_e.iterrows():
                res_dict[str(item)+" "+str(row[item])] = float(row['p'])/float(pr_e_p)

        print(res_dict)


if __name__ == "__main__":
    # Hardcoded voorbeeld om stuk te testen
    EigenBN = BNReasoner('climate_change.BIFXML')
    # print('MPE=', EigenBN.mpe(["Rainfall", "Deforestation", "Heatwaves"]))
    # print('MAP_variable_elim=', EigenBN.maximum_a_posteriori(["Rainfall", "Deforestation", "Heatwaves"]))
    # print('Prior marginal query=', EigenBN.marginal_distribution(["Rainfall", "Deforestation", "Heatwaves","Permafrost Melting", "Ice Melting"],False,False))
    print('Posterior marginal query=',EigenBN.marginal_distribution(["Rainfall"], "Heatwaves", True))
