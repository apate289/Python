class Solution:
    def expandAroundCenter(self, s, left, right):
        
        while left >= 0 and right < len(s) and s[left] == s[right]:
            print("left= {} right= {} s[left]= {}  s[right]= {}".format(left,right,s[left],s[right]))
            left -= 1
            right += 1
        return s[left + 1:right]

    def longestPalindrome(self, s):
        longest = ""
        
        for i in range(len(s)):
            print("------ Longest = {}  ----------".format(longest))
            print('----  odd  ------')
            odd = self.expandAroundCenter(s, i, i)
            print('odd = ',odd)
            print('----  even  -----')
            even = self.expandAroundCenter(s, i, i + 1)
            print('even = ',even)
            if len(odd) > len(longest):
                longest = odd
            if len(even) > len(longest):
                longest = even
        return longest


a= Solution()
long_string = a.longestPalindrome('abcdcdcb')
print(long_string)
