"""
Locks: Because Trusting Threads Is a Bad Idea
Question:
How do you make sure only one thread messes with your precious data at a time?

Answer: code given below

Explanation:
    Locks are like a “do not disturb” sign for your threads. 
    They make sure only one thread is messing with your data at a time, 
    which is good because threads are like toddlers — they get into everything.
"""

import threading

lock = threading.Lock()
counter = 0

def increment_counter():
    global counter
    with lock:
        for _ in range(10000):
            counter += 1  # Up, up, and away

threads = []
for _ in range(4):
    thread = threading.Thread(target=increment_counter)
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

print(counter)  # Hopefully, it's what you expect