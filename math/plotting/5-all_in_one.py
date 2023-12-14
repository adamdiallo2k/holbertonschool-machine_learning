#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Provided data
y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create figure and grid specification
fig = plt.figure(figsize=(10, 15))  # Adjust size as necessary
gs = gridspec.GridSpec(3, 2, figure=fig)
fig.suptitle('All in One', fontsize='large')

# First plot
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(y0, color='red')
ax0.set_title('Plot 1', fontsize='x-small')
ax0.set_xlabel('X', fontsize='x-small')
ax0.set_ylabel('Y', fontsize='x-small')

# Second plot
ax1 = fig.add_subplot(gs[0, 1])
ax1.scatter(x1, y1, color='magenta')
ax1.set_title("Men's Height vs Weight", fontsize='x-small')
ax1.set_xlabel('Height (in)', fontsize='x-small')
ax1.set_ylabel('Weight (lbs)', fontsize='x-small')

# Third plot
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(x2, y2, color='blue')
ax2.set_title('Exponential Decay of C-14', fontsize='x-small')
ax2.set_xlabel('Time (years)', fontsize='x-small')
ax2.set_ylabel('Fraction Remaining', fontsize='x-small')

# Fourth plot
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(x3, y31, label='t31')
ax3.plot(x3, y32, label='t32')
ax3.set_title('Decay of C-14 and Ra-226', fontsize='x-small')
ax3.set_xlabel('Time (years)', fontsize='x-small')
ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
ax3.legend(fontsize='x-small')

# Fifth plot
ax4 = fig.add_subplot(gs[2, :])
ax4.hist(student_grades, bins=10, edgecolor='black')
ax4.set_title('Project A', fontsize='x-small')
ax4.set_xlabel('Grades', fontsize='x-small')
ax4.set_ylabel('Number of Students', fontsize='x-small')

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
