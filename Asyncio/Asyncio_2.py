# How the heck do you use Python’s asyncio module for asynchronous tasks?
"""
Explaination:
So, asyncio is this thing that lets you run stuff concurrently, 
but not really at the same time. You’ll see async and await a lot, just remember, 
one without the other is like peanut butter without jelly.
"""
import asyncio

async def async_task():
    print("Task started")
    await asyncio.sleep(2)  # Because who doesn’t love waiting
    print("Task completed")

async def main():
    await asyncio.gather(async_task(), async_task())  # More tasks, more fun

asyncio.run(main())  # Let the chaos begin