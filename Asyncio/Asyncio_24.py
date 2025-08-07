"""
23. Context Managers: Now With Extra Safety
Question:
How do you create a context manager that handles exceptions like a boss?

Answer:

Explanation:
This context manager cleans up after itself and handles errors gracefully. 
It’s like having a personal assistant who also doubles as a bodyguard.
"""
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if exc_type:
            print(f"Exception occurred: {exc_val}")
            return True  # Swallow the error like a pro

with FileManager('Asyncio_24_TestFile.txt', 'w') as f:
    f.write('Hello, World!')
    raise ValueError("This is just a test, I swear")