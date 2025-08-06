"""
Deadlocks: Stuck in a Rut
Question:
What is a deadlock, and how can you avoid it before it ruins your day?

Answer:
Explanation:
    A deadlock happens when two (or more) threads are stuck waiting on each other 
    to release a resource. It’s like two people holding doors open for each other,
    neither one willing to go first. 
    Avoid it by planning your locks carefully or using a timeout to bail out.
"""
import threading
import time

# Define two shared resources (can be anything: files, DB connections, etc.)
lock_a = threading.Lock()
lock_b = threading.Lock()

def thread1():
    print("Thread 1: Acquiring lock A...")
    lock_a.acquire()
    print("Thread 1: Acquired lock A.")

    time.sleep(1)  # Simulate some work

    print("Thread 1: Waiting to acquire lock B...")
    lock_b.acquire()
    print("Thread 1: Acquired lock B.")

    print("Thread 1: Working with both locks...")
    lock_b.release()
    lock_a.release()

def thread2():
    print("Thread 2: Acquiring lock B...")
    lock_b.acquire()
    print("Thread 2: Acquired lock B.")

    time.sleep(1)  # Simulate some work

    print("Thread 2: Waiting to acquire lock A...")
    lock_a.acquire()
    print("Thread 2: Acquired lock A.")

    print("Thread 2: Working with both locks...")
    lock_a.release()
    lock_b.release()

# Create threads
t1 = threading.Thread(target=thread1)
t2 = threading.Thread(target=thread2)

# Start threads
t1.start()
t2.start()

# Wait for threads to finish (they won't because of deadlock)
t1.join()
t2.join()

print("Main thread finished.")
