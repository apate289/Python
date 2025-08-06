"""
Futures and Promises: Because Waiting Is the Hardest Part
Question:
What are Futures and Promises, and how do you use them in Python?

Answer: code given below

Explanation:
    Futures are like IOUs for results. You ask for something, go do other stuff, 
    and then come back when it’s ready. 
    Promises, well, they’re more of a JavaScript thing, but you get the idea.
"""
import concurrent.futures

def long_running_task():
    return "Task complete!"

with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(long_running_task)
    print(future.result())  # Was it worth the wait?