import re
zipcode = input()

print(bool(re.match(r'(?!.*(.).\1.*(.).\2)' r'(?!.*(.)(.)\3\4)' r'[1-9]\d{5}', zipcode)))
# First regex: checks a case of alternating repeating digits
# Second regex: Checks another case of alternating repeating digits (1212, etc)
# Checks structure: must start with 1-9 but can have 0s after


#==========================================================================
#
#==========================================================================
import re

# Regular expression to match integers in the range 100000 to 999999 inclusive
regex_integer_in_range = r"^[1-9][0-9]{5}$"  # Matches a 6-digit number starting with a non-zero digit.

# Regular expression to find alternating repetitive digit pairs
regex_alternating_repetitive_digit_pair = r"(?=(\d)\d\1)"  # Lookahead for digit patterns like 121, 232, etc.

# Input reading
P = input().strip()

# Validation: Check the postal code
is_valid = (bool(re.match(regex_integer_in_range, P)) and 
            len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)

# Output result
print(is_valid)



import re
P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)