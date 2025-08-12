"""
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);
 

Example 1:

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
"""

class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        
        if numRows == 1: return s
        a=""
        for i in range(numRows):
            print('--------   i:', i)
            for j in range(i,len(s),2*(numRows-1)):
                print(j,'s[j]:', s[j])
                a+=s[j]
                if(i>0 and i<numRows-1 and j+2*(numRows-1)-2*i < len(s)):
                    a+=s[j+2*(numRows-1)-2*i]
        return a
    
A = Solution()
print(A.convert("PAYPALISHIRING", 3))  
# Output: "PAHNAPLSIIGYIR" (3-1)*2 = 4 so need to collect letter at position 4
print(A.convert("PAYPALISHIRING", 4))  # Output: "PINALIGYPSH" (4-1)*2 = 6 so need to collect letter at position 6