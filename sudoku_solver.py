import os
import shutil
from functools import reduce
import sys
import random


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
            sudokus.append([0 if c == "." else int(c) for c in line])

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
                if val != 0:
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

def element_in_clause(clause, elements):
    for elem in elements:
        if elem in clause:
            return True
    return False


def evaluate_expression(clauses):
    """
    Evaluate an CNF expression.
    - If there's an empty clause, the expression is invalid.
    - If there's no clause left, the expression is valid
    """
    # Check if there's an empty clause.
    for clause in clauses:
        if len(clause) == 0:
            return False

    # There is a clause that isn't a literal.
    for clause in clauses:
        if len(clause) > 2:
            return 0

    return True


def get_variables(clauses):
    """
    Obtain all the positions that are in the CNF.
    """
    allPositions = set()
    for clause in clauses:
        for position in clause:
            allPositions.add(position)
    return sorted(allPositions)


def BCP(clauses):
    """
    Boolean Constraint Propagation
    - Every clause (other than the clause itself) containing l is removed (the clause is satisfied if l is).
    - In every clause that contains -l this literal is deleted (-l can not contribute to it being satisfied)
    """
    # Obtain literals that occur in the clauses
    literals = [clause for clause in clauses if len(clause) == 1]
    literals = reduce(lambda a, b: a.union(b), literals)  # Reduce list of sets
    already_neg_literals = set([literal for literal in literals if literal[0] == "-"])  # Negative literals
    neg_literals = set(["-" + literal for literal in literals if literal[0] != "-"]) # Negative literals
    neg_literals = neg_literals.union(already_neg_literals)

    new_clauses = []
    for clause in clauses:
        # Skip/remove clause if contains literal
        if element_in_clause(clause, literals):
            continue

        # Delete the literals that occur in the clause.
        if element_in_clause(clause, neg_literals):
            new_clauses.append(clause.difference(neg_literals))
        else:
            new_clauses.append(clause)

    return new_clauses


def DPLL(clauses, assignment, strategy):
    """
    DPLL algorithm, keep track of result so we can go back etc.

    DIMACS_file - Sudoku DIMACS file
    """
    new_clauses = BCP(clauses)
    evaluate_result = evaluate_expression(new_clauses)

    if evaluate_result is True:  # Formula evaluates to True -> return True
        return assignment
    elif evaluate_result is False:  # Formula evaluates to False -> return False
        return []
    else:  # Evaluate result returns 0 which means formula is undecided.
        # Choose next unassigned variable
        all_positions = set(get_variables(clauses))
        open_positions = sorted(all_positions.difference(assignment))
        choice = random.choice(open_positions)
        
        if strategy == '-S1':
            choice = random.choice(open_positions)
        if strategy == '-S2':
            choice = jw_os(open_positions, clauses)

        # Return (DPLL with that variable True) || (DPLL with that variable False)
        solution = None
        if choice[0] == "-":  # First positive choice
            new_clauses.append(set([choice[0:]]))
            solution = DPLL(new_clauses, assignment + [choice[0:]], strategy)
        else:
            new_clauses.append(set([choice]))
            solution = DPLL(new_clauses, assignment + [choice], strategy)

        if not solution:  # Negative choice
            if choice[0] == "-":  # Negative choice has been chosen
                new_clauses.append(set([choice]))
                solution = DPLL(new_clauses, assignment + [choice], strategy)
            else: # Make choice negative
                new_clauses.append(set(["-"+choice]))
                solution = DPLL(new_clauses, assignment + ["-"+choice], strategy)

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
    firstLine, clauses = data[0], data[1:]
    clauses = [set(clause.split()[:-1]) for clause in clauses]  # Remove zeroes

    # Perform Boolean Constraint Propagation
    assignment = []
    solution = DPLL(clauses, assignment, strategy)
    
    return solution


def jw_os(open_positions, clauses):
    ''' Decide which literal gets chosen during a split using Jeroslow-Wang
    open-positions - set of possible literals which can be split on
    clauses - set of clauses which these literals occur
    '''
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


if __name__ == "__main__":
    strategy_map = {
        "-S1": DPLL_Strategy,
    }

    # Parse the command line input
    strategy, file = parse_command()

    sudfile = "./testsets/4x4.txt"

    # Read sudoku from file.
    sudlist = read_sudoku(sudfile, 5)

    # Find N x N dimension for sudoku.
    sudlen = len(sudlist[0])
    N = 16 if sudlen == 256 else 9 if sudlen == 81 else 4

    # Read the problem (=clauses) as DIMACS file
    sudoku2DIMACS(sudlist, N, sudfile)

    # Implement DP + two heuristics
    nth_sudoku = 0
    parsedFile = sudfile.split("/")[-1].split(".")[0]
    DIMACS_file = "./sudoku_DIMACS/"+parsedFile + \
        "/"+parsedFile+"_"+str(nth_sudoku)+".cnf"
    
    # Strategy based on command line arguments
    result = strategy_map[strategy](DIMACS_file, strategy)
    print(result)


    # Write output (=variable assignments) as DIMACS file
