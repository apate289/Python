"""
You are given a string s and an array of strings words. All the strings of words are of the same length.
A concatenated string is a string that exactly contains all the strings of any permutation of words concatenated.
For example, if words = ["ab","cd","ef"], then "abcdef", "abefcd", "cdabef", "cdefab", "efabcd", and "efcdab" are all concatenated strings. "acdbef" is not a concatenated string because it is not the concatenation of any permutation of words.
Return an array of the starting indices of all the concatenated substrings in s. You can return the answer in any order.
 
Example 1:
	Input: s = "barfoothefoobarman", words = ["foo","bar"]
	Output: [0,9]
	Explanation:
	The substring starting at 0 is "barfoo". It is the concatenation of ["bar","foo"] which is a permutation of words.
	The substring starting at 9 is "foobar". It is the concatenation of ["foo","bar"] which is a permutation of words.
Example 2:
	Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
	Output: []
	Explanation:
	There is no concatenated substring.
Example 3:
	Input: s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
	Output: [6,9,12]
	Explanation:
	The substring starting at 6 is "foobarthe". It is the concatenation of ["foo","bar","the"].
	The substring starting at 9 is "barthefoo". It is the concatenation of ["bar","the","foo"].
	The substring starting at 12 is "thefoobar". It is the concatenation of ["the","foo","bar"].
Constraints:
1 <= s.length <= 104
1 <= words.length <= 5000
1 <= words[i].length <= 30
s and words[i] consist of lowercase English letters.

class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        word_l = len(words[0])
        word_c = len(words)
        word_length = word_l*word_c

        if len(s)<word_length:
            return []

        from collections import Counter
        word_counter = Counter(words)
        start = 0
        res = []

        for i in range(len(s)-word_length+1):
            window = s[i:i+word_length]
            word_length_counter = Counter()

            for j in range(0,word_length,word_l):
                word = window[j:j+word_l]
                word_length_counter[word]+=1
            
            if word_counter == word_length_counter:
                res.append(i)

        return res
        
"""

class Solution(object):
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        # all string words the same length
        ret = []
        n = len(s)
        k = len(words)
        m = len(words[0])
        p = m * k # total len of words
        initWindow = []
        notValid = set()
        valid = set()

        if n < p:
            # dont have enough words
            return []
            
        def validSubtring(substring):
            tempWords = words[:]
            i = 0
            while i < p:
                currWord = substring[i: i + m]
                if currWord in tempWords:
                    tempWords.remove(currWord)
                i += m
            
            return tempWords == []

        for i in range(p):
            letter = s[i]
            initWindow.append(letter)

        if validSubtring("".join(initWindow)):
            ret.append(0)

        i = p
        
        while i < n:
            letter = s[i]
            initWindow.pop(0)
            initWindow.append(letter)
            tempString = "".join(initWindow)
            # use caching here so no need to travsert again
            if tempString not in notValid:
                if tempString in valid or validSubtring(tempString):
                    ret.append(i - p + 1)
                    valid.add(tempString)
                else:
                    notValid.add(tempString)

            i += 1
        
        return ret
    
#----------------- fastest solution -----------------

class Solution_3(object):
# @param {string} s
# @param {string[]} words
# @return {integer[]}
    def findSubstring(self, s, words):
        if len(words) == 0:
            return []
        # initialize d, l, ans
        l = len(words[0])
        d = {}
        for w in words:   
            if w in d:
                d[w] += 1
            else:
                d[w] = 1
        i = 0
        print(d)
        ans = []

        # sliding window(s)
        for k in range(l):
            left = k
            print('left = ', left)
            subd = {}
            count = 0
            for j in range(k, len(s)-l+1, l): # (0, 18-3+1, 3) for 1st example
                tword = s[j:j+l]
                print('j =', j, 'tword =', tword)
                # valid word
                if tword in d:
                    #print(tword)
                    if tword in subd:
                        subd[tword] += 1
                    else:
                        subd[tword] = 1
                    count += 1
                    while subd[tword] > d[tword]:
                        subd[s[left:left+l]] -= 1
                        left += l
                        count -= 1
                    if count == len(words):
                        print('left = ', left, 'count = ', count)
                        ans.append(left)
                # not valid
                else:
                    left = j + l
                    subd = {}
                    count = 0

        return ans
	
x= Solution_3()
print(x.findSubstring("barfoothefoobarman", ["foo","bar"]))  # Output: [0,9]
#print(x.findSubstring("wordgoodgoodgoodbestword", ["word","good","best","word"]))  # Output: []
#print(x.findSubstring("barfoofoobarthefoobarman", ["bar","foo","the"]))  # Output: [6,9,12]
