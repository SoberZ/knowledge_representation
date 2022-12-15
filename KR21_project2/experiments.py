import os
# OpenMP problem fix. Remove if it causes problems.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from BNReasoner import BNReasoner
import timeit


def time_me(fun, *args):
    start_time = timeit.default_timer()
    res = fun(*args)
    diff = timeit.default_timer() - start_time
    return diff

reasoner = BNReasoner('testing/dog_problem.BIFXML')

#####################################################
# d-separation: DFS vs BFS
# X, Y, Z = ["dog-out"], ["light-on"], ["hear-bark"]
# n_times = 1
# avg1, avg2 = 0, 0
# for _ in range(0, n_times):
#     avg1 += time_me(reasoner.d_separation, X, Y, Z)
#     avg2 += time_me(reasoner.d_separation_BFS, X, Y, Z)

# print("Avg DFS time: ", avg1/n_times)
# print("Avg BFS time: ", avg2/n_times)

# Avg DFS time:  0.0006145293499999705
# Avg BFS time:  0.0005937194500000187


#####################################################
# Variable elimination vs naive summing out
X = ["dog-out"]
reasoner.bn.draw_structure()
n_times = 20
avg1, avg2 = 0, 0
for _ in range(0, n_times):
    avg1 += time_me(reasoner.maximum_a_posteriori, X)
    avg2 += time_me(reasoner.maximum_a_posteriori_marginalize, X)

print("Avg VE time: ", avg1/n_times)
print("Avg SO time: ", avg2/n_times)


###############################
# unpruned vs pruned

