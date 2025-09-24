
def maxRepeating(seq: str, word: str) -> int:
    print(seq)
    print(word)
    if word not in seq or (not word):
        print('returning...')
        return 0
    
    n = len(seq)//len(word)
    print('n = ', n)
    ans = 0
    for i in range(1,n+1):
        print('i = ',i)
        if word*i in seq:
            print(word*i)
            ans = max(ans, i)
    #return ans
    print(ans)


maxRepeating("ababc", word= "ab")
print('---------------')
maxRepeating("ababc", word="ba")
print('---------------')
maxRepeating("ababc", word="ac")
print('---------------')
maxRepeating("ababc", word="de")
print('---------------')
maxRepeating("ababc", word="")
print('---------------')
#maxRepeating("ababc", word=12)
print('---------------')
maxRepeating("ababc", word="a1")
