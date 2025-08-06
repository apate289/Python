# Avoiding DeadLock with Threading

import threading
import time

# Shared resources
lock_a = threading.Lock()
lock_b = threading.Lock()

def thread1():
    print("Thread 1: Acquiring lock A...")
    with lock_a:
        print("Thread 1: Acquired lock A.")
        time.sleep(1)

        print("Thread 1: Acquiring lock B...")
        with lock_b:
            print("Thread 1: Acquired lock B.")
            print("Thread 1: Working with both locks...")

def thread2():
    print("Thread 2: Acquiring lock A...")
    with lock_a:  # Changed from lock_b to lock_a to keep order same as thread1
        print("Thread 2: Acquired lock A.")
        time.sleep(1)

        print("Thread 2: Acquiring lock B...")
        with lock_b:
            print("Thread 2: Acquired lock B.")
            print("Thread 2: Working with both locks...")

# Start threads
t1 = threading.Thread(target=thread1)
t2 = threading.Thread(target=thread2)

t1.start()
t2.start()

t1.join()
t2.join()

print("Main thread finished — no deadlock!")
