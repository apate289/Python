"""
19. Metaclasses: When Regular Classes Just Don’t Cut It
Question:
What are metaclasses in Python, and how do you use them?

Answer:

Explanation:
Metaclasses are like classes for classes. They let you define how your classes behave, 
but using them is like jumping down the rabbit hole. Be careful.
"""

class Meta(type):
    def __new__(cls, name, bases, dct):
        print(f'Creating class {name}')
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):
    pass

obj = MyClass()