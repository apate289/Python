"""
20. Handling Multiple Exceptions: Because One Isn’t Enough
Question:
    How can you handle multiple exceptions in a single except block?

Answer:

Explanation:
    Python lets you handle multiple exceptions in one go by passing a tuple of exceptions. 
    It’s efficient, and who doesn’t love efficiency?
"""

try:
    # Code that might fail spectacularly
    pass
except (TypeError, ValueError) as e:
    print(f"An error occurred: {e}")