#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
x = np.arange(0, 11)

# Tracé avec modification de la couleur et du style de la ligne
plt.plot(x, y, color='red', linestyle='-')  # Ligne solide rouge

plt.xlabel('Axe X')
plt.ylabel('Axe Y')
plt.title('Graphique avec ligne solide rouge')

plt.show()