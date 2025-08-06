"""
16. Producer-Consumer Problem: The Endless Loop
Question:
How do you solve the Producer-Consumer problem using Python’s Queue?
Explanation:
The Producer-Consumer problem is a classic in concurrency land. Queue makes it easier, 
but don’t be fooled—it’s still tricky business.
"""
import threading
from queue import Queue

def producer(q):
    for i in range(5):
        q.put(i)
        print(f'Produced {i}')
    q.put(None)  # End of line, folks

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f'Consumed {item}')
        q.task_done()

q = Queue()
threading.Thread(target=producer, args=(q,)).start()
threading.Thread(target=consumer, args=(q,)).start()