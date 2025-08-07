"""
18. LRU Cache: Forgetfulness with a Purpose
Question:
How do you implement a Least Recently Used (LRU) cache in Python?

Answer:


Explanation:
An LRU cache throws out the least recently used items to make room for new ones. It’s like spring cleaning for your data.
"""
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Bye-bye, oldest entry

cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # Returns 1
cache.put(3, 3)
print(cache.get(2))  # Returns -1 because it’s been evicted