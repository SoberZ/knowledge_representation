import shutil
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


def sudoku2DIMACS(sudokus, N, sudfile):
    """
    Convert a sudou to DIMACS format. We use the given sudokus
    and append the values in the sudoku to it to generate a newly
    created sudoku in DIMACS format.

    sudokus - list of sudokus
    N - N x N dimension sudoku
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
        newfile = filetowrite+"_"+str(index)+".cnf"

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


if __name__ == "__main__":
    # strategy_map = {
    #     "S1": strategy1,
    # }

    # Parse the command line input
    # strategy, file = parse_commmand()

    sudfile = "./testsets/4x4.txt"

    # Read sudoku from file.
    sudlist = read_sudoku(sudfile, 1)

    # Find N x N dimension for sudoku.
    sudlen = len(sudlist[0])
    N = 16 if sudlen == 256 else 9 if sudlen == 81 else 4

    # Read the problem (=clauses) as DIMACS file
    sud2dimacs = sudoku2DIMACS(sudlist, N, sudfile)

    # Implement DP + two heuristics

    # Write output (=variable assignments) as DIMACS file
