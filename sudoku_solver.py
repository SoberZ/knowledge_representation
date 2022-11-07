import sys


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


def parse_commmand():
    """
    Parse the input from the command line.
    Input format: python3 sudoku_solver.py -Sn file
    """

    strategy = "-S1"
    file = "./example_sudokus/4x4.txt"

    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        file = sys.argv[2] if len(sys.argv > 2) else file

    return strategy, file


if __name__ == "__main__":
    # strategy_map = {
    #     "S1": strategy1,
    # }

    # Parse the command line input
    # strategy, file = parse_commmand()

    # Read sudoku from file.
    sud = read_sudoku("./testsets/4x4.txt", 1)

    # Read the problem (=clauses) as DIMACS file
    
    # Implement DP + two heuristics

    # Write output (=variable assignments) as DIMACS file

