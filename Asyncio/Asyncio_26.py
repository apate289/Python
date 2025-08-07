"""
25. Decorators on Classes: Because Functions Aren’t Enough
Question:
How do you apply a decorator to a class?

Answer:

Explanation:
Decorators aren’t just for functions anymore. 
Use them on classes to add new methods or change behavior, 
but remember, with great power comes great confusion.
"""

def decorator(cls):
    class NewClass(cls):
        def new_method(self):
            return "I'm new here!"

    return NewClass

@decorator
class MyClass:
    def original_method(self):
        return "I'm the OG"

obj = MyClass()
print(obj.original_method())  # Still works
print(obj.new_method())  # Surprise! I'm new here!