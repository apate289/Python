import re
N = int(input())
'''
A valid UID must follow the rules below:

It must contain at least 2 uppercase English alphabet characters.
It must contain at least 3 digits (0 - 9).
It should only contain alphanumeric characters (a - z , A - Z & 0 - 9).
No character should repeat.
There must be exactly 10 characters in a valid UID.
'''
# Declare the patterns
two_plus_uppercase = r'(?=(?:.*[A-Z]+){2,})'
three_plus_digits = r'(?=(?:.*\d){3,})'
no_repeats = r'(?!.*(.).*\1)' 
ten_alphanumeric = r'[\w]{10}'
filters = [no_repeats, two_plus_uppercase, three_plus_digits, ten_alphanumeric]

for i in range(0, N):
    uid = input()
    print('Valid') if all([re.match(f, uid) for f in filters]) else print('Invalid')