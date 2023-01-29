# Python3 code to demonstrate
# smallest number greater than K
# using sort() + bisect_right()
# https://www.geeksforgeeks.org/python-find-smallest-element-greater-than-k/
from bisect import bisect_right

# Initializing list (dictionary keys)
test_dict = {1:'a', 4: 'b', 7: 'c', 5: 'd', 10: 'e'}

# Initializing k
k = 4

# Need to sort keys for bisect operation
sorted_keys = sorted(test_dict.keys())

# Printing original list
print("The input dictionary is " + str(test_dict))
print("The keys in original order are : " + str(test_dict.keys()))
print("The sorted keys are : " + str(sorted_keys))

# Using sorted() + bisect_right()
# to find smallest number greater than k
# and its position
index = bisect_right(sorted_keys, k)
min_val = sorted_keys[index]

# Printing result
print("The minimum key value greater than " + str(k) + " is : " + str(min_val) + " at index position " + str(index))
print("The value associated with that key is " + test_dict[min_val])
