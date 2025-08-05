#Thread Pools: Because Who Has Time?
#How do you set up a thread pool in Python to get stuff done faster? Or slower, who knows?
"""
Explanation:Thread pools are supposed to make things run concurrently, 
but they also make you question everything you know about GIL (more on that later).
"""

from concurrent.futures import ThreadPoolExecutor

def task(name):
    print(f"Task {name} is running")  # Because running is fun, right?

with ThreadPoolExecutor(max_workers=4) as executor:
    names = ['A', 'B', 'C', 'D']
    executor.map(task, names)  # It's like magic, but with threads