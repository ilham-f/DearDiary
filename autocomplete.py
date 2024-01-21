from transformers import GPT2LMHeadModel, GPT2Tokenizer
import string

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

class AutocompleteAIWithNLP(AutocompleteAI):
    def __init__(self, model_name='gpt2', max_length=50):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def generate_context(self, user_input):
        input_ids = self.tokenizer.encode(user_input, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=self.max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

# Example usage with context understanding:

autocomplete_ai_with_nlp = AutocompleteAIWithNLP()

file_path = 'indonesian-words.txt'
with open(file_path, 'r') as txt_file:
    lines = txt_file.readlines()

lines = [line.strip() for line in lines]
indonesian_words = lines

for word in indonesian_words:
    autocomplete_ai_with_nlp.insert(word)

# Get input from the user
user_input = input("Masukkan kalimat: ")

# Generate context using NLP
context = autocomplete_ai_with_nlp.generate_context(user_input)
print(f"Context: {context}")

# Extract the last word from the context and get autocomplete suggestions
words = context.split()
if words:
    # Remove punctuation from the last word
    last_word = words[-1].translate(str.maketrans('', '', string.punctuation))
    
    suggestions = autocomplete_ai_with_nlp.search(last_word)
    print(f"Autocomplete suggestions for '{last_word}': {suggestions}")
else:
    print("No words in the input context.")
