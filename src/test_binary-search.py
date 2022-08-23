# Python3 code to demonstrate
# smallest number greater than K
# using sort() + bisect_right()
# https://www.geeksforgeeks.org/python-find-smallest-element-greater-than-k/
from bisect import bisect_right

# Initializing list (dictionary keys)
test_dict = {1:'a', 4: 'b', 7: 'c', 5: 'd', 10: 'e'}

# Initializing k
k = 12

# Printing original list
print("The original list is : " + str(test_dict.keys()))

# Using sorted() + bisect_right()
# to find smallest number
# greater than k
sorted_keys = sorted(test_dict.keys())
min_val = sorted_keys[bisect_right(sorted_keys, k)]

# Printing result
print("The minimum value greater than " + str(k) + " is : " + str(min_val))