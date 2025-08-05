# Asyncio + HTTP Requests = ???
# How do you handle multiple HTTP requests asynchronously using asyncio and aiohttp? 
# Because who needs simplicity?
"""
Explanation:
aiohttp is what you use when you want to make your life harder by doing HTTP requests asynchronously. 
The example above? Yeah, it works...most of the time.
"""
import asyncio
import aiohttp 

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()  # Maybe it works, maybe it doesn’t

async def main():
    async with aiohttp.ClientSession() as session:
        urls = ['https://example.com', 'https://python.org', 'https://github.com']
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        for result in results:
            print(result)  # Or print an error, who knows

asyncio.run(main())  # Fingers crossed