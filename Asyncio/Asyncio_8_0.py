# Decorators with Arguments: Extra Fancy
#Question:How do you make a decorator that takes arguments,
#  like it wasn’t already complicated enough?
"""
Explanation:
Decorators are already tricky, but add arguments and now you’ve got yourself a real party.
 It’s like inception but with functions.
"""

def repeat(times):
    def decorator(func): # func becomes "say_hello"
        def wrapper(*args, **kwargs): 
            #wrapper(*args, **kwargs) means:"Define a wrapper function that can accept 
            # and pass through any arguments to the original function, 
            # no matter how many or what kind."
            #say_hello("") gets passed into wrapper(*args, **kwargs) as:
            #args = ("",)
            #kwargs = {}
            print(*args)
            for _ in range(times):
                func(*args, **kwargs)  # Do it again, and again, and again...
        return wrapper
    return decorator

@repeat(3)  # Why just once when you can do it three times?
def say_hello():
    print("Hello!")

say_hello()