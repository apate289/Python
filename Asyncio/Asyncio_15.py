"""
14. Decorators: Now with Extra Parameters!
Question:
How do you make a decorator that takes multiple arguments? Because one argument is never enough.

Answer: code given below
Explanation:
Decorators can be made to accept parameters by defining a function that returns a decorator.Decorators can do a lot more when you let them take arguments, 
like logging or repeating actions. Just be careful not to get too carried away.
"""

def repeat_and_log(times, message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                print(message)
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat_and_log(3, "Calling function")
def say_hello():
    print("Hello!")

say_hello()