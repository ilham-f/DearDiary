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