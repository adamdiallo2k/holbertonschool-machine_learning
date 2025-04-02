#!/usr/bin/env python3
"""
Creates a pd.DataFrame from a np.ndarray
"""
import numpy as np
import pandas as pd
import string

def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray
    
    Parameters:
        array (np.ndarray): The NumPy array to convert
        
    Returns:
        pd.DataFrame: The newly created DataFrame with alphabetical column labels
    """
    # Get number of columns in the array
    _, num_cols = array.shape
    
    # Create column labels (A, B, C, ...) based on number of columns
    # Ensure we don't exceed 26 columns (as per requirements)
    column_labels = [string.ascii_uppercase[i] for i in range(min(num_cols, 26))]
    
    # Create DataFrame with the array and column labels
    df = pd.DataFrame(array, columns=column_labels)
    
    return df

# Exemple d'utilisation :
if __name__ == "__main__":
    # Création d'un array NumPy de démonstration
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    
    # Conversion en DataFrame
    df_result = from_numpy(data)
    print(df_result)
