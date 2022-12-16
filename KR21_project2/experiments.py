import os
# OpenMP problem fix. Remove if it causes problems.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from BNReasoner import BNReasoner
import timeit
import networkx as nx
import statistics



def time_me(fun, *args):
    start_time = timeit.default_timer()
    res = fun(*args)
    diff = timeit.default_timer() - start_time
    return diff

reasoner = BNReasoner('climate_change.BIFXML')

#####################################################
# d-separation vs d-separation pruning
# X, Y, Z = ["dog-out"], ["light-on"], ["hear-bark"]
# n_times = 50
# avg1vals, avg2vals = [], []
# for _ in range(0, n_times):
#     avg1vals.append(time_me(reasoner.d_separation, X, Y, Z))
#     avg2vals.append(time_me(nx.d_separated, reasoner.bn.structure, set(X), set(Y), set(Z)))

# print("Avg normal time: ", statistics.mean(avg1vals))
# print("STD normal time: ", statistics.stdev(avg1vals))

# print("Avg pruning time: ", statistics.mean(avg2vals))
# print("STD pruning time: ", statistics.stdev(avg2vals))


# Avg DFS time:  0.0006145293499999705
# Avg BFS time:  0.0005937194500000187


#####################################################
# Variable elimination vs naive summing out
X = ["Earth Surface Reflectivity"]
n_times = 3
avg1vals, avg2vals = [], []
for _ in range(0, n_times):
    avg1vals.append(time_me(reasoner.maximum_a_posteriori, X))
    avg2vals.append(time_me(reasoner.maximum_a_posteriori_marginalize, X))

print("Avg VE time: ", statistics.mean(avg1vals))
print("STD VE time: ", statistics.stdev(avg1vals))

print("Avg SO time: ", statistics.mean(avg2vals))
print("STD SO time: ", statistics.stdev(avg2vals))

###############################
# unpruned vs pruned

