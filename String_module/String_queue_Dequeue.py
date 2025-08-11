"""
Approach
✅ Approach: Greedy with Queues (Deque)
We simulate the voting round using 
two queues (implemented via Deque), one for each party:

r for Radiant senators (index positions).

d for Dire senators (index positions).

🔁 Logic:

1. Traverse the input senate string and add each index i to the corresponding queue:

If character is 'R', add i to r.

If character is 'D', add i to d.

2. While both queues are not empty, simulate one round of voting:

Compare the front of both queues (r.peek() and d.peek()).

The senator with the smaller index acts first and bans the other.

The acting senator gets re-added with 
    a new index i + n (to simulate coming back in the next round).

Remove both current senators from the front.

3. Once one queue becomes empty, the party with remaining senators wins.

Complexity
Time complexity: O(N)
Space complexity: O(N)
Code
"""
from collections import deque

class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        r = deque()
        d = deque()
        n = len(senate)

        for i in range(n):
            if senate[i] == 'R':
                r.append(i)
            else:
                d.append(i)

        while r and d:
            r_idx = r.popleft()
            d_idx = d.popleft()

            if r_idx < d_idx:
                r.append(r_idx + n)
            else:
                d.append(d_idx + n)

        return "Radiant" if r else "Dire"
a= Solution()
print(a.predictPartyVictory("RD"))  # Example usage
