"""
Problem 4: Max Length Subarray with Sum 0 (LC 325-style)
MEDIUM

Find the maximum length subarray with sum equal to zero (or a target k). Uses a hash map storing the first occurrence of each prefix sum.

Approach
11.	Store the first index where each prefix sum occurs.
12.	When the same prefix sum recurs at index i, the subarray between first occurrence and i sums to 0.
13.	Storing first (not latest) occurrence maximizes the subarray length.

Output log:
curr =  1 Num =  1 Index =  0
prefix_idx =  {0: -1, 1: 0}
curr =  0 Num =  -1 Index =  1
prefix_idx[curr] =  -1  Max Len =  0
Updated Max Len =  2
curr =  3 Num =  3 Index =  2
prefix_idx =  {0: -1, 1: 0, 3: 2}
curr =  0 Num =  -3 Index =  3
prefix_idx[curr] =  -1  Max Len =  2
Updated Max Len =  4
curr =  2 Num =  2 Index =  4
prefix_idx =  {0: -1, 1: 0, 3: 2, 2: 4}
curr =  2 Num =  0 Index =  5
prefix_idx[curr] =  4  Max Len =  4
Updated Max Len =  4
curr =  0 Num =  -2 Index =  6
prefix_idx[curr] =  -1  Max Len =  4
Updated Max Len =  7
curr =  2 Num =  2 Index =  7
prefix_idx[curr] =  4  Max Len =  7
Updated Max Len =  7
7
"""

def maxLenSubarrayZeroSum(nums):
    prefix_idx = {0: -1}   # sum -> first index seen
    curr = 0
    max_len = 0
 
    for i, num in enumerate(nums):
        curr += num
        #print ("curr = ", curr, "Num = ", num, "Index = ", i)
        if curr in prefix_idx:
            #print("prefix_idx[curr] = ", prefix_idx[curr], " Max Len = ", max_len)
            max_len = max(max_len, i - prefix_idx[curr])
            #print("Updated Max Len = ", max_len)
        else:
            prefix_idx[curr] = i   # store FIRST occurrence only
            #print("prefix_idx = ", prefix_idx)
 
    return max_len
 
# Example
nums = [1, -1, 3, -3, 2, 0, -2, 2]
print(maxLenSubarrayZeroSum(nums))   # Output: 8 (entire array)
