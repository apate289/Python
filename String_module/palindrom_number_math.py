import os,sys

def palindrome_number_math(x):
    if x < 0 or (x % 10 == 0 and x != 0):
        return False
    rev = 0
    while x > rev:
        print(x,rev)
        rev = rev * 10 + x % 10
        x //= 10
    print('x == rev', x==rev)
    print('x == rev // 10', x==rev // 10)
    return x == rev or x == rev // 10


a = palindrome_number_math(123)
print(a)

a = palindrome_number_math(121)
print(a)

a = palindrome_number_math(1213)
print(a)

