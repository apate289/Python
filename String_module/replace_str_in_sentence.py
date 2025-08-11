class Solution:
    def replaceWords(self, dictionary, sentence):
        sentence_list = sentence.split(' ')
        for i in range(len(sentence_list)):
            for word in dictionary:
                if word== sentence_list[i][:len(word)]: 
                    sentence_list[i]= word
        return (' '.join(word for word in sentence_list))

a = Solution()
print(a.replaceWords(["cat", "bat", "rat"], "the cattle was rattled by the battery"))  # Example usage


"""
----------------------------------------------------------------------------------
# Fastest solution using Trie for replacing words in a sentence with their roots:
----------------------------------------------------------------------------------
from typing import List

class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None  # store full root word at end node

class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        # Build Trie from dictionary
        root = TrieNode()
        for word in dictionary:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word  # mark the end of a root word
        
        # Replace words in sentence
        def find_root(word):
            node = root
            for char in word:
                if char not in node.children:
                    return word  # no prefix found
                node = node.children[char]
                if node.word:
                    return node.word  # return the shortest matching root
            return word
        
        return ' '.join(find_root(w) for w in sentence.split())

"""