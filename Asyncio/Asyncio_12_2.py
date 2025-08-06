import threading
import time

# Shared resources
lock_a = threading.Lock()
lock_b = threading.Lock()

def thread1():
    print("Thread 1: Trying to acquire lock A...")
    if lock_a.acquire(timeout=2):
        print("Thread 1: Acquired lock A.")
        try:
            time.sleep(1)
            print("Thread 1: Trying to acquire lock B...")
            if lock_b.acquire(timeout=2):
                try:
                    print("Thread 1: Acquired lock B. Working...")
                finally:
                    lock_b.release()
            else:
                print("Thread 1: Could not acquire lock B. Avoiding deadlock.")
        finally:
            lock_a.release()
    else:
        print("Thread 1: Could not acquire lock A.")

def thread2():
    print("Thread 2: Trying to acquire lock B...")
    if lock_b.acquire(timeout=2):
        print("Thread 2: Acquired lock B.")
        try:
            time.sleep(1)
            print("Thread 2: Trying to acquire lock A...")
            if lock_a.acquire(timeout=2):
                try:
                    print("Thread 2: Acquired lock A. Working...")
                finally:
                    lock_a.release()
            else:
                print("Thread 2: Could not acquire lock A. Avoiding deadlock.")
        finally:
            lock_b.release()
    else:
        print("Thread 2: Could not acquire lock B.")

# Start threads
t1 = threading.Thread(target=thread1)
t2 = threading.Thread(target=thread2)

t1.start()
t2.start()

t1.join()
t2.join()

print("Main thread finished — no deadlock due to timeout handling!")
