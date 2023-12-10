import numpy as np
import matplotlib.pyplot as plt

# Data for the x-axis
x = np.linspace(0, 20000, 400)  # 400 points from 0 to 20,000

# Constants for decay
half_life_C14 = 5730       # Half-life of C-14 is 5730 years
half_life_Ra226 = 1600     # Half-life of Ra-226 is 1600 years

# Decay rate calculation
decay_rate_C14 = np.log(0.5) / half_life_C14
decay_rate_Ra226 = np.log(0.5) / half_life_Ra226

# Exponential decay functions
y1 = np.exp(decay_rate_C14 * x)  # Decay of C-14
y2 = np.exp(decay_rate_Ra226 * x)  # Decay of Ra-226

# Creating the plot
plt.plot(x, y1, 'r--', label='C-14')  # Dashed red line for C-14
plt.plot(x, y2, 'g-', label='Ra-226')  # Solid green line for Ra-226

# Setting the title, labels, and axis limits
plt.title("Exponential Decay of Radioactive Elements")
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.xlim(0, 20000)
plt.ylim(0, 1)

# Adding the legend
plt.legend(loc='upper right')

# Showing the plot
plt.show()
