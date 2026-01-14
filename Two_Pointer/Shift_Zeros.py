import os

#Input: nums = [0, 1, 0, 3, 2]
#Output: [1, 3, 2, 0, 0]

nums = [0, 1, 0, 3, 2]
nums = [0,0,1]
zeros_list = []
left = 0

for i in nums:
    if(i == 0):
        zeros_list.append(i)
        nums.remove(i)

print(nums)
print(zeros_list)
nums.extend(zeros_list)
print(nums)

nums = [0,0,1]
for right in range(len(nums)):
    if nums[right] != 0:
        nums[left], nums[right] = nums[right], nums[left]
        # Increment 'left' since it now points to a position already occupied
        # by a non-zero element.
        left += 1

print(nums)
