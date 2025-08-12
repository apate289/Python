"""
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', 
determine if the input string is valid.
An input string is valid if:  Open brackets must be closed by the same type of brackets.
	Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
Example 1:
	Input: s = "()"
	Output: true
Example 2:
	Input: s = "()[]{}"
	Output: true
Example 3:
	Input: s = "(]"
	Output: false
Example 4:
	Input: s = "([])"
	Output: true
Example 5:
	Input: s = "([)]"
	Output: false
 
Constraints:
1 <= s.length <= 104
s consists of parentheses only '()[]{}'.
"""
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # Initialize an empty stack and a hash map for matching brackets
        stack = []
        hash = {')': '(', ']': '[', '}': '{'}
        
        # Loop through each character in the string
        for char in s:
            if char in hash:
                # If it's a closing bracket, check the stack
                if stack and stack[-1] == hash[char]:
                    stack.pop()  # Remove the matching opening bracket
                else:
                    return False  # Invalid if no match
            else:
                # Push opening brackets onto the stack
                stack.append(char)
        
        # Return True if stack is empty, False otherwise
        return not stack

x= Solution()
print(x.isValid("()"))          # Output: True
print(x.isValid("()[]{}"))      # Output: True
print(x.isValid("(]"))          # Output: False
print(x.isValid("([)]"))        # Output: False
print(x.isValid("{[]}"))       # Output: True