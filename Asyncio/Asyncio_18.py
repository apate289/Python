"""
17. Asyncio Semaphore: Because You Can’t Do Everything at Once
Question:
How do you use a Semaphore in asyncio to limit the number of concurrent tasks?

Answer: Code given below
Explanation:
    A Semaphore in asyncio is like the velvet rope at a nightclub. 
    Only a certain number of tasks can get in at once, keeping things under control.
"""

import asyncio

async def limited_task(sem, n):
    async with sem:
        print(f'Task {n} started')
        await asyncio.sleep(1)
        print(f'Task {n} finished')

async def main():
    sem = asyncio.Semaphore(2)  # Only 2 at a time, folks
    tasks = [limited_task(sem, i) for i in range(5)]
    await asyncio.gather(*tasks)

asyncio.run(main())