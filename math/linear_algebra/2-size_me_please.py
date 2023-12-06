#!/usr/bin/env python3

def compute_shape_3d(nested_list):
    if not isinstance(nested_list, list) or not nested_list:
        return []

    # Get the length of the outer list
    outer_len = len(nested_list)

    # Initialize dimensions for the inner lists
    inner_len = 0
    inner_inner_len = 0

    # Iterate over each item in the outer list
    for item in nested_list:
        # Check if the item is a list itself
        if isinstance(item, list):
            inner_len = max(inner_len, len(item))
            # Iterate over each item in the inner list
            for inner_item in item:
                if isinstance(inner_item, list):
                    inner_inner_len = max(inner_inner_len, len(inner_item))

    # Return the shape as a tuple
    return (outer_len, inner_len, inner_inner_len) if inner_inner_len else (outer_len, inner_len)

# Example usage
matrix_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
print("Shape:", compute_shape_3d(matrix_3d))  # Output will be (3, 2, 2)
