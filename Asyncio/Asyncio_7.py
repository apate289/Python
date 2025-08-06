# Generators: Lazy or Genius?
# Question:How do you create and use a generator in Python without pulling your hair out?

"""
Explanation:
Generators are like the lazy people of Python. They only do work when you ask them to, 
which is cool because they don’t use up all your memory.
"""
def simple_generator():
    yield 1
    yield 2
    yield 3

gen = simple_generator()
for value in gen:
    print(value)  # Look, Ma! No memory!