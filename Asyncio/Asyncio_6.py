# Singletons: Because We Only Need One
# How do you make sure you only ever have one instance of a class in Python? Like, ever?

"""
Explanation:
Singletons are great when you want to make sure you only have one instance of a class, 
but it’s also a good way to confuse people reading your code.
"""
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance

singleton1 = Singleton()
singleton2 = Singleton()

print(singleton1 is singleton2)  # True, hopefullycls
