import numpy as np
import nltk
from nltk import pos_tag
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data for POS tagging
nltk.download('averaged_perceptron_tagger')

# Read the content of the text file
file_path = 'indonesian_words.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    indonesian_words = [word.strip() for word in file.readlines()]

# Tokenize the Indonesian words dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(indonesian_words)
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
input_sequences = []
for line in indonesian_words:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))

# Build the LSTM model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_length-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, batch_size=128, epochs=10, verbose=1, shuffle=True)

# Function to get POS of a word
def get_pos(word):
    return pos_tag([word])[0][1]

# Function to generate the next word with priority based on specific POS tags
def generate_next_word_with_priority(seed_text, model, tokenizer, max_sequence_length, desired_pos_tags):
    seed_text_lower = seed_text.lower()
    token_list = tokenizer.texts_to_sequences([seed_text_lower])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    
    # Get the POS of the last word in the seed text
    last_word_pos = get_pos(seed_text_lower.split()[-1])

    # Generate a range of suggestions
    suggestions = []
    max_attempts = 10 # Maximum number of attempts to generate suggestions
    for _ in range(max_attempts):
        predictions = model.predict(token_list)
        
        # Add some randomness to encourage diversity
        predicted_index = np.random.choice(len(predictions[0]), p=predictions[0])
        predicted_word = tokenizer.index_word[predicted_index]
        predicted_word_pos = get_pos(predicted_word)
        
        # Exclude the seed_text from suggestions
        if predicted_word != seed_text_lower and predicted_word_pos in desired_pos_tags:
            suggestions.append(predicted_word)

    # If no suitable suggestions are found, consider a wider range
    if not suggestions:
        for _ in range(10):  # Generate 10 suggestions without filtering by POS
            predictions = model.predict(token_list)
            predicted_index = np.random.choice(len(predictions[0]), p=predictions[0])
            predicted_word = tokenizer.index_word[predicted_index]
            suggestions.append(predicted_word)

    generated_text = " ".join(suggestions)
    return generated_text

# Example usage
seed_text = "saya mau ma"
desired_pos_tags = ['VB', 'VBG', 'VBP', 'VBZ']  # Desired POS tags for verbs
predicted_text = generate_next_word_with_priority(seed_text, model, tokenizer, max_sequence_length, desired_pos_tags)
print(f"Suggestions for '{seed_text}': {predicted_text}")
