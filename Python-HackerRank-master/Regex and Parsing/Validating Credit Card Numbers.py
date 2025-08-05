import re

N = int(input())
# Starts with 4 or 5 or 6, consists of either 4 groups of 4 (split by a hyphen) or no groups at all
structure_check = r'[456]\d{3}(-?\d{4}){3}$'
# No digit repeats more than 4 times
no_four_repeats = r'((\d)-?(?!(-?\2){3})){16}'
filters = structure_check, no_four_repeats
for i in range(0, N):
    cc_num = input()
    print('Valid') if all(re.match(f, cc_num) for f in filters) else print('Invalid')
    
    
#---------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------------------------

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
t = int(input())
p1 = r'^[456]'
p2 = r'^.{16,19}$'
p3 = r'\d{4}-\d{4}-\d{4}-\d{4}'
p4 = r'[^@_!#$%^&*()<>?/\|}{~:]'
p5 = r'(\d)(-\1+)'
p6 = r'(\d)\1{3,}'

for i in range(t):
    number = input()
    c1 = re.match(p1,number) 
    c2 = re.match(p2,number)
    c3= re.search(p3,number)
    c4 = re.match(p4,number)
    c5 = re.search(p5,number)
    c6 = re.search(p6,number)
    #print(c1,c2,c3,c4,c5,c6)
    if '-' in number:
        if c1 != None and c2 != None and c3 != None and c4 != None and c6 == None and c5 == None:
            print('Valid')
        else:
            print('Invalid')
    else:
        if c1 != None and c2 != None and c4 != None:
            print('Valid')
        else:
            print('Invalid')