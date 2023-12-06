#!/usr/bin/env python3
"""commented module"""



def matrix_shape(matrix):
    def get_shape_recursive(current_element):
        if not isinstance(current_element, list):
            return []
        return [len(current_element)] + get_shape_recursive(current_element[0])

    return get_shape_recursive(matrix)
