# How to use the Sudoku Solver

To execute the python3 file: *python3 sudoku_solver.py **-Sn** **testfile***

- For **-Sn** - *n* can be a number from {1,2,3}

Strategies that have been implemented:

1. -S1: **Random Choice**
2. -S2: **JW-OS**
3. -S3: **MOMS**


testfile: the file that contains a (list of) sudoku(s).

If you want to change the number of sudokus you want to read from a file set **n_sudokus** to a number that you would like. If **n_sudokus** is -1 then all sudokus from the file will be read.

