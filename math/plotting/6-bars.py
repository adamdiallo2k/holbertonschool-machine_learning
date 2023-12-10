#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Assign colors to each fruit
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruit_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']

# Names of people
people = ['Farrah', 'Fred', 'Felicia']

# Creating the stacked bar graph
bar_width = 0.5
r = np.arange(len(people))
bottom = np.zeros(len(people))

# Plotting each fruit
for i in range(fruit.shape[0]):
    plt.bar(r, fruit[i, :], bar_width, bottom=bottom, color=colors[i], label=fruit_labels[i])
    bottom += fruit[i, :]

# Adding labels and title
plt.xlabel('Person')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(r, people)
plt.yticks(np.arange(0, 81, 10))

# Adding the legend
plt.legend()

# Display the plot
plt.show()


