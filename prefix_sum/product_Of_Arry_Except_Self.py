"""
Problem 5: Product of Array Except Self (LeetCode 238)
MEDIUM

Return an array where output[i] equals the product of all elements except nums[i]. Division is not allowed. Uses prefix and suffix product passes.

Approach
14.	Left pass: result[i] = product of all elements to the left of i.
15.	Right pass: multiply result[i] by product of all elements to the right of i.
16.	Two O(n) passes replace any need for division.

result_Pre =  [1, 1, 1, 1]
prefix =  1
result_Pre =  [1, 1, 1, 1]
prefix =  2
result_Pre =  [1, 1, 2, 1]
prefix =  6
result_Pre =  [1, 1, 2, 6]
prefix =  24
result_Suf =  [1, 1, 2, 6]
suffix =  4
result_Suf =  [1, 1, 8, 6]
suffix =  12
result_Suf =  [1, 12, 8, 6]
suffix =  24
result_Suf =  [24, 12, 8, 6]
suffix =  24
[24, 12, 8, 6]

Output log:
[24, 12, 8, 6]
"""
def productExceptSelf(nums):
    n = len(nums)
    result = [1] * n
 
    # Left pass: result[i] = product of nums[0..i-1]
    prefix = 1
    for i in range(n):
        result[i] = prefix
        print("result_Pre = ", result)
        prefix *= nums[i]
        print("prefix = ", prefix)
 
    # Right pass: multiply by product of nums[i+1..n-1]
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        print("result_Suf = ", result)
        suffix *= nums[i]
        print("suffix = ", suffix)
 
    return result
 
# Example
print(productExceptSelf([1, 2, 3, 4]))
# Output: [24, 12, 8, 6]
