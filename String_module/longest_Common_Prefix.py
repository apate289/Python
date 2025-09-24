
"""
def longestCommonPrefix(self, strs: List[str]) -> str:
    pref = strs[0]
    pref_len = len(pref)

    for s in strs[1:]:
        while pref != s[0:pref_len]:
            pref_len -= 1
            if pref_len == 0:
                return ""
            
            pref = pref[0:pref_len]
    
    return pref
"""
#class Solution:
#   def longestCommonPrefix(self, strs: List[str]) -> str:

def longest_common_prefix(strs):
    #print(strs)
    ans=""
    v=sorted(strs)
    #print(v)
    first=v[0]
    last=v[-1]
    #print(first, last)
    #print(min(len(first),len(last)))
    for i in range(min(len(first),len(last))):
        if(first[i]!=last[i]):
            return ans
        ans+=first[i]
    return ans


a = longest_common_prefix(['flower','flow','flaunt'])
print(a)
'''print('---------------------------')
a = longest_common_prefix(["dog","racecar","car"])
print(a)
print('---------------------------')
a = longest_common_prefix(['flower','flow','flaunt',""])
print(a)
print('---------------------------')
a = longest_common_prefix(["dog","dograce","dogcar"])
print(a)
'''
