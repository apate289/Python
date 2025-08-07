class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s)==1:
            return 1
        d={}
        l,r = 0,0
        n = len(s) #"pwwkew"
        maxLen = 0
        while(r < n):
            print(f'--------r = {r}--------')
            if s[r] in d:
                l = max(l,d[s[r]]+1)
                print(f'l = {l}')
            length = r - l +1
            print(f'length = {length}')
            maxLen = max(maxLen,length)
            print(f'maxLen = {maxLen}')
            d[s[r]]=r
            print(d)
            r+=1
        return maxLen
        