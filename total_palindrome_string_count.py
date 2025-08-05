class Solution(object):

    def expandOddAroundCenter(self, s, i, n): 
        j,oC = 0,0
        while j < n:
            if i-j < 0 or i+j >= n or s[i-j] != s[i+j]:
                break
            oC += 1
            j += 1
        return oC
        
    def expandEvenAroundCenter(self, s, i, n): 
        j,eC = 0,0
        while j < n:
            if i-j < 0 or i+j+1 >= n or s[i-j] != s[i+j+1]:
                break
            eC += 1
            j += 1
        return eC

    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        longest = "" 
        count = 0 
        n = len(s)
        for i in range(len(s)):
            odd = self.expandOddAroundCenter(s, i, n)
            even = self.expandEvenAroundCenter(s, i, n)
            count += (odd + even)
        return count


s = Solution()
subCnt = s.countSubstrings('abcdcaba')