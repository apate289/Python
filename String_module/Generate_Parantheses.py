"""
Given n pairs of parentheses, write a function to generate 
all combinations of well-formed parentheses.
Example 1:
    Input: n = 3
    Output: ["((()))","(()())","(())()","()(())","()()()"]
Example 2:
    Input: n = 1
    Output: ["()"]
Constraints:
1 <= n <= 8
"""
class Solution_B(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        ans = []
        queue = [('(', 1,0)]  # Item will contain (str, open_nos, close_nos)
        f = 0             # front_index
        l = 1            # queue_length
        while f<l:
            cur = queue[f]
            if cur[1]+cur[2]==2*n:
                ans.append(cur[0])
                f+=1
                continue
            if cur[1] < n:
                queue.append((cur[0]+'(', cur[1]+1, cur[2]))
                l+=1
            if cur[2] < cur[1]:
                queue.append((cur[0]+')', cur[1], cur[2]+1))
                l+=1
            f+=1
        return ans
        

#----------  Fastest Solution ----------------------
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        def dfs(left, right, s):
            if len(s) == n * 2:
                res.append(s)
                return 

            if left < n:
                dfs(left + 1, right, s + '(')

            if right < left:
                dfs(left, right + 1, s + ')')

        res = []
        dfs(0, 0, '')
        return res

x = Solution()
print(x.generateParenthesis(3))
print(x.generateParenthesis(1))

x = Solution_B()
print(x.generateParenthesis(3))
print(x.generateParenthesis(1))