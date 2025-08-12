"""
Given a string containing just the characters '(' and ')', 
return the length of the longest valid (well-formed) parentheses substring.
Example 1:
	Input: s = "(()"
	Output: 2
	Explanation: The longest valid parentheses substring is "()".
Example 2:
	Input: s = ")()())"
	Output: 4
	Explanation: The longest valid parentheses substring is "()()".
Example 3:
	Input: s = ""
	Output: 0
 
Constraints:
0 <= s.length <= 3 * 104
s[i] is '(', or ')'.

class Solution(object):
    def longestValidParentheses(self, s):
        stack=[]
        l=['0']*len(s)
        for ind,i in enumerate(s):
            if i=='(':
                stack.append(ind)
            else:
                if stack:
                    l[stack.pop()]='1'
                    l[ind]='1'
        return max(len(i) for i in ''.join(l).split('0'))
        
        # Return True if stack is empty, False otherwise
        return not stack

"""

class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        l=[-1]
        c=0
        for i,ch in enumerate(s):
            if ch=="(":
                l.append(i)
            else:
                l.pop()
                if not l:
                    l.append(i)
                else:
                    c=max(c,i-l[-1])
        return c

x= Solution()
print(x.longestValidParentheses("(()"))          # Output: 2
print(x.longestValidParentheses(")()())"))       # Output: 4
print(x.longestValidParentheses(""))              # Output: 0
print(x.longestValidParentheses("()(()"))        # Output: 2