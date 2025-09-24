#file reading by reverse line
print('------  file reading by reverse line  ------')
with open('file_reading.txt', 'r') as file:
    lines = file.readlines()
    for line in reversed(lines):
        print(line.strip())

#file reading by reverse character
print('------  file reading by reverse character  ------')
with open('file_reading.txt', 'r') as file:
    content = file.read()
    print(content[::-1])
