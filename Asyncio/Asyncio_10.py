"""
Asyncio Queues: Why Should Threads Have All the Fun?
Question:
    How do you manage tasks in an async program without everything falling apart?
Explanation:
    asyncio.Queue is your best friend when you need to keep track of tasks in an
    async environment. It’s like Queue, but for people who love await.
"""
import asyncio

async def worker(name, queue):
    while True:
        task = await queue.get()
        if task is None:
            break
        print(f'{name} processing task {task}')
        await asyncio.sleep(1)  # Take your time, no rush
        queue.task_done()

async def main():
    queue = asyncio.Queue()

    workers = [asyncio.create_task(worker(f'worker-{i}', queue)) for i in range(3)]

    for i in range(10):
        await queue.put(i)

    await queue.join()

    for w in workers:
        await queue.put(None)

    await asyncio.gather(*workers)

asyncio.run(main())