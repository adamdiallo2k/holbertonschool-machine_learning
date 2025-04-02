#!/usr/bin/env python3
"""
Creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray

    Parameters:
        array: The NumPy array to convert (assumes it's already defined
        externally)

    Returns:
        pd.DataFrame: The newly created DataFrame with alphabetical column
        labels
    """
    # Get number of columns in the array
    _, num_cols = array.shape

    # Alphabet en dur, pour générer les labels de colonnes
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
 
    # Génère les labels pour les colonnes (A, B, C, ...)
    # en se limitant à 26 caractères max
    column_labels = list(alphabet[:min(num_cols, 26)])

    # Crée le DataFrame avec les labels de colonne
    df = pd.DataFrame(array, columns=column_labels)

    return df


# Exemple d'utilisation (à titre de test seulement) :
if __name__ == "__main__":
    # Supposons que 'data' soit un np.ndarray déjà défini ailleurs ;
    # on montre ici un tableau d'exemple en pur Python pour la démo :
    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    # Conversion en DataFrame
    df_result = from_numpy(data)
    print(df_result)
