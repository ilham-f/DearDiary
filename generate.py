import random

# Read vocabulary from a text file
vocabulary_file_path = 'indonesian_words.txt'  # Replace with the actual file path
with open(vocabulary_file_path, 'r', encoding='utf-8') as vocab_file:
    vocabulary = [word.strip() for word in vocab_file]

# Generate 1000 random sentences
sentences = []
for _ in range(10000):
    sentence_length = random.randint(5, 15)  # Random sentence length between 5 and 15 words
    sentence = " ".join(random.sample(vocabulary, sentence_length))
    sentences.append(sentence.capitalize() + ".")

# Write sentences to a file
output_file_path = 'indonesia_diary_sentences.txt'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write('\n'.join(sentences))

print(f"File '{output_file_path}' has been created with 1000 randomly generated diary-like sentences.")
