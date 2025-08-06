# Thread-Safe Queues: Because Safety First
# Question: How do you make sure your threads play nice and don’t stomp all over each other’s data?
# Answer: By using thread-safe queues! They help manage the data flow between threads without conflicts.
"""
Explanation: Using a Queue is like having a bouncer at the club; 
it keeps everything orderly so the threads don’t get out of hand.
"""
from queue import Queue
from threading import Thread

def worker(q):
    while not q.empty():
        item = q.get()
        print(f"Processing {item}")  # Do work, get paid, repeat
        q.task_done()

q = Queue()
for i in range(10):
    q.put(i)  # Fill 'er up

threads = []
for _ in range(4):
    thread = Thread(target=worker, args=(q,))
    print(f"Starting thread {thread.name}")  # Debugging print
    thread.start()
    threads.append(thread)

q.join()
for thread in threads:
    print(f"Thread {thread.name} is alive after completion: {thread.is_alive()}") # Debugging print
    thread.join()
