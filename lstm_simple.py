import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

# Membaca file teks
file_path = "kalimat_diary.txt"
with open(file_path, "r") as file:
    diary_text = file.read()

# Memisahkan teks menjadi kalimat-kalimat
diary_sentences = diary_text.split("\n")

# Tokenisasi kalimat-kalimat dalam dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(diary_sentences)
vocab_size = len(tokenizer.word_index) + 1

# Menghasilkan urutan token dari setiap kalimat
sequences = []
for line in diary_sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)

# Padding urutan token untuk membuat input sequences sepanjang yang terpanjang
max_sequence_len = max([len(seq) for seq in sequences])
sequences = np.array(pad_sequences(sequences, maxlen=max_sequence_len, padding='pre'))

# Membagi input dan output sequences
X, y = sequences[:,:-1],sequences[:,-1]
y = np.eye(vocab_size)[y]

# Membangun model LSTM
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=2)
model.save('lstm_autocomplete.keras')

# with open('tokenizer.pkl', 'wb') as f:
#     pickle.dump(tokenizer, f)

# Fungsi untuk menghasilkan teks berikutnya berdasarkan teks input
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted = np.argmax(predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Contoh penggunaan model untuk menghasilkan teks berikutnya
seed_text = "Saya merasa"
next_words = 5
generated_text = generate_text(seed_text, next_words, model, max_sequence_len)
print(generated_text)
