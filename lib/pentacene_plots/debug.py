import numpy as np

# Example array
original_array = np.array([
    [1, 2, 5],
    [4, 5, -3],
    [7, 8, -8]
])

# Get indices to sort by decreasing absolute value of the third column
indices = np.argsort(np.abs(original_array[:, 2]))[::-1]

# Use the indices to rearrange the entire array
sorted_array = original_array[indices]

print("Original Array:")
print(original_array)
print("\nSorted Array by Decreasing Absolute Value of the Third Column:")
print(sorted_array)
