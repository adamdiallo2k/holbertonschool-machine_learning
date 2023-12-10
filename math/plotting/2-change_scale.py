import numpy as np
import matplotlib.pyplot as plt

# Generating the data
x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# Creating the plot
plt.plot(x, y, color='red')  # Solid red line
plt.yscale('log')

# Setting the title and labels
plt.title("Exponential Decay of C-14")
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')

# Setting x-axis limits
plt.xlim(0, 28650)

# Showing the plot
plt.show()
