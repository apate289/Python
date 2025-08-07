"""
21. Catching and Re-Raising Exceptions: Throwing It Back
Question:
How do you catch and re-raise an exception in Python?

Answer:

Explanation:
Sometimes, you catch an exception just to log it or clean up, 
but you still want it to go up the chain. That’s where comes in handy.
"""

try:
    # Something that will definitely break
    raise ValueError("Oops, something went wrong")
except ValueError as e:
    print(f"Caught an exception: {e}")
    raise  # Pass it along for someone else to deal with