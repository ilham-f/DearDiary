class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class AutocompleteAI:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = []
        self._dfs(node, prefix, suggestions)
        return suggestions

    def _dfs(self, node, current_prefix, suggestions):
        if node.is_end_of_word:
            suggestions.append(current_prefix)

        for char, child_node in node.children.items():
            self._dfs(child_node, current_prefix + char, suggestions)


# Example usage:
autocomplete_ai = AutocompleteAI()

file_path = 'indonesian_words.txt'

# Read the content of the text file and store each line as a string in a list
with open(file_path, 'r') as txt_file:
    lines = txt_file.readlines()

# Strip leading and trailing whitespaces from each line
lines = [line.strip() for line in lines]

# print(lines)

# Assuming you have a dataset of Indonesian words
indonesian_words = lines 

for word in indonesian_words:
    autocomplete_ai.insert(word)

# Get prefix input from the user
prefix_to_search = input("Masukkan kalimat: ")
# suggestions = autocomplete_ai.search(prefix_to_search)

# Split the sentence into words and get the last word
words = prefix_to_search.split()
if words:
    last_word = words[-1]
    suggestions = autocomplete_ai.search(last_word)
    print(f"Autocomplete suggestions for '{last_word}': {suggestions}")
else:
    print("No words in the input sentence.")