import os
import shutil
from functools import reduce
import sys
import random
import time

def read_sudoku(file, n_lines=-1):
    """
    Read n sudokus from a file.

    file - input sudoku file
    n_lines - read n sudokus from the file
              -1 means read all sudokus
    """

    sudokus = []
    with open(file) as f:
        for line in f.readlines():
            if n_lines != -1:
                if n_lines == 0:
                    return sudokus
                n_lines -= 1

            line = line.replace("\n", "")
            sudokus.append(['0' if c == "." else c for c in line])

    return sudokus


def parse_command():
    """
    Parse the input from the command line.
    Input format: python3 sudoku_solver.py -Sn file
    """

    strategy = "-S1"
    file = "./example_sudokus/4x4.txt"

    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        file = sys.argv[2] if len(sys.argv) > 2 else file

    return strategy, file


def sudoku2DIMACS(sudokus, N, sudfile):
    """
    Convert a sudoku to DIMACS format. We use the given sudokus
    and append the values in the sudoku to it to generate a newly
    created sudoku in DIMACS format.

    sudokus - list of sudokus
    N - N x N dimension sudoku
    sudfile - sudoku file that is used
    """

    # Rules mapping to find the rules corresponding to the N.
    N2rules = {
        4: "./sudoku_rules/sudoku-rules-4x4.txt",
        9: "./sudoku_rules/sudoku-rules-9x9.txt",
        16: "./sudoku_rules/sudoku-rules-16x16.txt"
    }

    # Obtain all the values for the sudoku.
    for index, sudoku in enumerate(sudokus):
        values = []
        for i in range(1, N+1):
            for j in range(1, N+1):
                val = sudoku[(i-1)*N+(j-1)]  # index is r*N+c
                if val != '0':
                    values.append(str(i)+str(j)+str(val))

        # Create the new .cnf file for the sudoku.
        filetowrite = sudfile.split("/")[-1].split(".")[0]
        dimacDir = "./sudoku_DIMACS/" + filetowrite

        if not os.path.exists(dimacDir):
            os.makedirs(dimacDir)

        newfile = dimacDir+"/"+filetowrite+"_"+str(index)+".cnf"

        # Copy the rules from one file to the new file.
        shutil.copyfile(N2rules[N], newfile)

        # Increment number of clauses in the first line.
        first_line = ""
        file_data = []
        with open(newfile, "r") as f:
            file_data = f.readlines()
            first_line = file_data[0].split()
            first_line[-1] = str(int(first_line[-1]) + len(values))
            first_line = ' '.join(first_line) + "\n"

        # Replace first line.
        file_data[0] = first_line
        with open(newfile, "w") as f:
            f.writelines(file_data)

        # Write values to file
        with open(newfile, "a+") as f:
            for value in values:
                f.write(value+" 0\n")


def get_variables(clauses):
    """
    Obtain all the positions that are in the CNF.
    """
    allPositions = []
    for clause in clauses:
        for position in clause:
            allPositions += [position]
    return allPositions


def negate_unit(unit):
    """
    Negation of a string unit.
    """
    return unit[1:] if unit[0] == "-" else "-" + unit


def single_BCP(clauses, unit):
    new_clauses = []
    for clause in clauses:
        # Whole clause is true, therefore delete
        if unit in clause:
            continue

        # Negative of the unit is in clause, remove unit
        negation_unit = negate_unit(unit)
        if negation_unit in clause:
            new_clause = [x for x in clause if x != negation_unit]
            if new_clause == []:
                return -1
            new_clauses.append(new_clause)
        else:
            new_clauses.append(clause)
    return new_clauses


def unit_propagation(clauses):
    assignment = []
    new_clauses = []
    unit_clauses = [clause for clause in clauses if len(clause) == 1]
    for unit in unit_clauses:
        new_clauses = single_BCP(clauses, unit[0])
        assignment.append(unit[0])
        if new_clauses == -1:
            return new_clauses, assignment
        if not new_clauses:
            return -1, []
        unit_clauses = [clause for clause in new_clauses if len(clause) == 1]

    return new_clauses, assignment


def DPLL(clauses, assignment, strategy):
    """
    DPLL algorithm, keep track of result so we can go back etc.

    DIMACS_file - Sudoku DIMACS file
    """
    global n_propagations
    n_propagations += 1
    print(n_propagations)

    new_clauses, new_assign = unit_propagation(clauses)
    assignment += new_assign

    if new_clauses == -1:
        global n_backtrack
        n_backtrack += 1
        return []
    if new_clauses == []:
        return assignment

    all_positions = get_variables(new_clauses)

    # Choose next unassigned variable
    if strategy == '-S2':
        choice = jw_os(all_positions, new_clauses)
    elif strategy == '-S3':
        choice = moms(all_positions, new_clauses)
    else: # strategy == '-S1' or default
        choice = random.choice(all_positions)

    # Return (DPLL with that variable True) || (DPLL with that variable False)
    solution = DPLL(new_clauses + [[choice]], assignment + [choice], strategy)
    if not solution:
        negation_choice = negate_unit(choice)
        solution = DPLL(new_clauses + [[negation_choice]], assignment + [negation_choice], strategy)

    return solution


def DPLL_Strategy(DIMACS_file, strategy):
    """
    DPLL algorithm, keep track of result so we can go back etc.

    DIMACS_file - Sudoku DIMACS file
    """
    data = []
    with open(DIMACS_file, "r") as f:
        data = f.readlines()

    # First line contains N variables and N clauses
    _, clauses = data[0], data[1:]
    new_clauses = []
    for clause in clauses:
        splitted = clause.split()[:-1]  # Remove zeroes
        new_clause = []
        for split in splitted:
            new_clause.append(split)
        new_clauses.append(new_clause)

    # Perform Boolean Constraint Propagation
    assignment = []
    start_time = time.time()
    solution = DPLL(new_clauses, assignment, strategy)
    end_time = time.time()

    global time_lapsed
    time_lapsed += end_time - start_time

    return solution


def jw_os(open_positions, clauses):
    """  Decide which literal gets chosen during a split using Jeroslow-Wang
    open-positions - set of possible literals which can be split on
    clauses - set of clauses which these literals occur
    """
    # Keep track of the highest J value calculated for any literal
    max_j_value = 0
    # Create a variable to return the selected literal with
    selected_literal = open_positions[0]
    # Loop over all literals in the set open_positions to decide which literal has the highest J value
    if len(open_positions) > 0:
        for literal in open_positions:
            j_value = 0
            for clause in clauses:
                if literal in clause:
                    j_value += 2 ** (-abs(len(clause)))
            if j_value >= max_j_value:
                max_j_value = j_value
                selected_literal = literal

    return selected_literal


def moms(open_positions, clauses):
    """Decide which literal gets chosen during a split using MOM's heuristic
    open-positions - set of possible literals which can be split on
    clauses - set of clauses which these literals occur"""
    # Initiate used variables
    selected_literal = open_positions[0]
    max_score = 0
    min_clause_length = min(clauses, key=lambda x: len(x))
    min_clauses = []
    moms_score = 0
    # Loop over all clauses, find minimal clause length and add those clauses to a list
    for clause in clauses:
        if len(clause) == min_clause_length:
            min_clauses.append(clause)
    # Loop over all literals in the open_positions to calculate moms score per literal
    for literal in open_positions:
        k = 1
        literal_count = 0
        neg_literal_count = 0
        # Count how many times the current literal occurs in the smallest clauses
        for min_clause in min_clauses:
            if str(literal) in min_clause:
                literal_count += 1
            if str(literal*-1) in min_clause:
                neg_literal_count += 1

        # Calculate the MOM formula per literal to see if its value is the largest, if so, select it
        moms_score = ((literal_count + neg_literal_count) * 2 **
                      k) + (literal_count*neg_literal_count)
        if moms_score > max_score:
            max_score = moms_score
            selected_literal = literal

    return selected_literal


def sudoku_print(sudoku, N):
    """Use function to print sudoku's with, easier visible to check
    sudoku - the sudoku with positions as '111' format which means value 1 on row 1 column 1"""
    print_sudoku = []
    for i in range(N):
        row = []
        print_sudoku.append(row)
        for j in range(N):
            row.append('0')

    for position in sudoku:
        position = str(position)
        if len(position) == 3:
            if print_sudoku[int(position[0])-1][int(position[1])-1] not in ('0', str(position[2])):
                print('FOUT IN SUDOKU!! MEERDERE ASSIGNMENTS VOOR DEZELDE PLEKKEN')
            print_sudoku[int(position[0])-1][int(position[1])-1] = position[2]
    for row in print_sudoku:
        print(row)


if __name__ == "__main__":
    # Parse the command line input
    strategy, sudfile = parse_command()

    # Read sudoku from file.
    n_sudokus = 10
    sudlist = read_sudoku(sudfile, n_sudokus)

    # Find N x N dimension for sudoku.
    sudlen = len(sudlist[0])
    N = 16 if sudlen == 256 else 9 if sudlen == 81 else 4

    # Read the problem (=clauses) as DIMACS file
    sudoku2DIMACS(sudlist, N, sudfile)

    total_backtrack = 0
    n_backtrack = 0
    time_lapsed = 0
    n_propagations = 0

    # Implement DP + two heuristics
    for i in range(0, n_sudokus):
        n_backtrack = 0
        n_propagations = 0

        parsedFile = sudfile.split("/")[-1].split(".")[0]
        DIMACS_file = "./sudoku_DIMACS/"+parsedFile + \
            "/"+parsedFile+"_"+str(i)+".cnf"
        result = DPLL_Strategy(DIMACS_file, strategy)
        # sudoku_print(result, N)
        print("N backtrack: ", n_backtrack)
        total_backtrack += n_backtrack

    print("total time: ", time_lapsed)
    print("avg time: ", time_lapsed/n_sudokus)
    print("avg backtracks: ", total_backtrack/n_sudokus)
    print("avg evaluations: ", n_propagations/n_sudokus)

    # Write output (=variable assignments) as DIMACS file
