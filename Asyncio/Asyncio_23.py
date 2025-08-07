"""
22. Sorting Complicated Data: Because Simple Is Overrated
Question:
How do you sort a list of dictionaries by a specific key?

Answer:

Explanation:
Sorting a list of dictionaries is as easy as using a lambda. 
Just don’t get too carried away with complicated keys.
"""
data = [
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25},
    {'name': 'Charlie', 'age': 35}
]

sorted_data = sorted(data, key=lambda x: x['age'])
print(sorted_data)  # Now in age order