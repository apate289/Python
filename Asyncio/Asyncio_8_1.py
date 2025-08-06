"""
2. Why use wrapper(*args, **kwargs) in a decorator?
When writing decorators, the wrapper function replaces the original function, so it needs to be flexible enough to handle any arguments that the original function might have received.

This is especially important for asynchronous functions (async def) in asyncio — because:

You may be decorating async functions (which return coroutines),

And the wrapper itself might also need to be async def to await them.

wrapper(*args, **kwargs) makes sure the decorator works with any function signature.

await func(*args, **kwargs) calls the original async function.

The decorator adds timing functionality while preserving original behavior.
"""
import asyncio
import time
from functools import wraps

def async_timer(func):
    print('now in async_timer...') #1st print
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        print('now in wrapper...') #3rd print
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds") #6th print
        print('now returning from wrapper...') #7th print
        return result
    print('now returning from async_timer...') #2nd print
    return wrapper

@async_timer
async def my_async_function(x):
    print('here.. 1') #4th print
    await asyncio.sleep(x)
    print('here.. 2') #5th print
    return x

# Run it
asyncio.run(my_async_function(2))