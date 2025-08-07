"""
24. Immutable Classes: Lock It Down
Question:
How do you create an immutable class in Python that doesn’t let anyone mess with its attributes?

Answer:

Explanation:
Immutable classes are like Fort Knox for your data. 
Once set, no one can change them, which is sometimes just what you need.
"""
class ImmutablePoint:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

point = ImmutablePoint(1, 2)
print(point.x)  # 1
print(point.y)  # 2
# point.x = 10  # Uncommenting this will make Python angry