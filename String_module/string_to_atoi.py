"""
String to Integer (atoi):
Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer.

The algorithm for myAtoi(string s) is as follows:

Whitespace: Ignore any leading whitespace (" ").
Signedness: Determine the sign by checking if the next character is '-' or '+', assuming positivity if neither present.
Conversion: Read the integer by skipping leading zeros until a non-digit character is encountered or the end of the string is reached. If no digits were read, then the result is 0.
Rounding: If the integer is out of the 32-bit signed integer range [-231, 231 - 1], then round the integer to remain in the range. Specifically, integers less than -231 should be rounded to -231, and integers greater than 231 - 1 should be rounded to 231 - 1.
Return the integer as the final result.
Example 1:
    Input: s = "42"
    Output: 42
    Explanation:
        The underlined characters are what is read in and the caret is the current reader position.
        Step 1: "42" (no characters read because there is no leading whitespace)
                ^
        Step 2: "42" (no characters read because there is neither a '-' nor '+')
                ^
        Step 3: "42" ("42" is read in)
            ^
Example 2:
    Input: s = " -042"
    Output: -42
    Explanation:
        Step 1: "   -042" (leading whitespace is read and ignored)
                    ^
        Step 2: "   -042" ('-' is read, so the result should be negative)
                    ^
        Step 3: "   -042" ("042" is read in, leading zeros ignored in the result)
                    ^
Example 3:
    Input: s = "1337c0d3"
    Output: 1337
    Explanation:
        Step 1: "1337c0d3" (no characters read because there is no leading whitespace)
                ^
        Step 2: "1337c0d3" (no characters read because there is neither a '-' nor '+')
                ^
        Step 3: "1337c0d3" ("1337" is read in; reading stops because the next character is a non-digit)
                    ^
Example 4:
    Input: s = "0-1"
    Output: 0
    Explanation:
        Step 1: "0-1" (no characters read because there is no leading whitespace)
                ^
        Step 2: "0-1" (no characters read because there is neither a '-' nor '+')
                ^
        Step 3: "0-1" ("0" is read in; reading stops because the next character is a non-digit)
                ^
Example 5:
    Input: s = "words and 987"
    Output: 0
    Explanation:
        Reading stops at the first non-digit character 'w'.



Constraints:

0 <= s.length <= 200
s consists of English letters (lower-case and upper-case), digits (0-9), ' ', '+', '-', and '.'.

---------------------------
0 ms response time
class Solution(object):
    def myAtoi(self, s):
        result = '0'
        s = s.strip()
        if s == '':
            return 0
        if s[0] == '-':
            for i in range(1, len(s)):
                if (not s[i].isdigit()):
                    break
                else:
                    result += s[i]
            result = "-"+result
        else:
            for i in range(len(s)):
                if s[i] == '+' and i == 0:
                    pass
                elif (not s[i].isdigit()):
                    break
                else:
                    result += s[i]
        try:
            if int(result) < -2**31:
                return -2**31
            elif (2**31)-1 < int(result):
                return (2**31)-1
        except:
            pass
        return int(result)
---------------------------
"""

class Solution(object):
    
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = s.strip()
        if not s:
            return 0
        i = 0
        sign = 1
        result = 0
        if s[i] == '-' or s[i] == '+':
            sign = -1 if s[i] == '-' else 1
            i += 1
        while i < len(s) and s[i].isdigit():
            digit = int(s[i])
            if result > (2**31 - 1 - digit) // 10:
                return 2**31 - 1 if sign == 1 else -2**31
            result = result * 10 + digit
            print(result, 'result after adding digit:', digit)
            i += 1
        return sign * result
    

x = Solution()
print(x.myAtoi("42"))          # Output: 42
print(x.myAtoi("   -42"))      # Output: -42
print(x.myAtoi("4193 with words"))  # Output: 4193
print(x.myAtoi("words and 987"))    # Output: 0
print(x.myAtoi("-91283472332"))     # Output: -2147483648
print(x.myAtoi("487 and 987"))     # Output: 487