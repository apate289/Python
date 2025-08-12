"""

"""
class Solution_1(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        return haystack.find(needle)
        

class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        n = len(haystack)
        m = len(needle)

        if m > n:
            return -1

        i = 0
        j = 0

        while i < n:
            if haystack[i] == needle[j]:
                i += 1
                j += 1
                if j == m:
                    return i - m
            else:
                i = i - j + 1
                j = 0

        return -1

#---------- Fastest Class ----------#        
a= Solution_1()
print(a.strStr("hello", "ll"))  # Output: 2
print(a.strStr("aaaaa", "bba"))  # Output: -1
print(a.strStr("", ""))  # Output: 0
