"""
Context Managers: Because with is Classy
Question:
How do you make your code look cool and handle resources properly with the with statement?

Answer: Code given below

Explanation:
    Context managers are there to make sure you clean up your messes. 
    The with statement is like having someone else do the dishes after you cook.
"""

class FileManager:
    def __init__(self, filename, mode):
        self.file = open(filename, mode)

    def __enter__(self):
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()  # Clean up after yourself

with FileManager('Asyncio_13_TestFile.txt', 'w') as f:
    f.write("Hello, World!")  # Write like a boss