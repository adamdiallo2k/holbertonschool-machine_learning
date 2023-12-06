#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = [element for row in matrix for element in (row[2], row[3])]
print("The middle columns of the matrix are: {}".format(the_middle))

