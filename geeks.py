import re 
import string 
import torch 
import pandas as pd 
from docx import Document 
from collections import Counter
from torch import nn 

# Read the DOCX file 
doc_path = "wikipedia.docx"
doc = Document(doc_path) 

# Extract text from paragraphs 
text_data = [paragraph.text for paragraph in doc.paragraphs] 

# Convert text to lowercase 
text_data =  text_data = [text.lower() for text in text_data]

# Remove special characters and words between them using regex 
text_data = [re.sub(r"\[.*?\]", "", text) for text in text_data] 

# Remove words not in the English alphabet 
english_alphabet = set(string.ascii_lowercase)
text_data = [' '.join([word for word in text.split() if all(char in english_alphabet for char in word)]) for text in text_data]

# Remove leading/trailing whitespaces and empty sentences
text_data = [sentence.strip() for sentence in text_data if sentence.strip()]

# Create a DataFrame with the cleaned text data 
df = pd.DataFrame({"Text": text_data}) 

# Save the cleaned text data to a CSV file 
output_path = "output.csv"
# Set index=False to exclude the index column in the output 
df.to_csv(output_path, index=False) 

print("Text data cleaned and saved to:", output_path) 

class TextDataset(torch.utils.data.Dataset): 
	def __init__(self, args): 
		self.args = args 
		self.words = self.load_words() 
		self.unique_words = self.get_unique_words() 

		self.index_to_word = {index: word for index,
							word in enumerate(self.unique_words)} 
		self.word_to_index = {word: index for index,
							word in enumerate(self.unique_words)} 

		self.word_indexes = [self.word_to_index[w] for w in self.words] 

	def load_words(self): 
		train_df = pd.read_csv('output.csv') 
		text = train_df['Text'].str.cat(sep=' ') 
		return text.split(' ') 

	def get_unique_words(self): 
		word_counts = Counter(self.words) 
		return sorted(word_counts, key=word_counts.get, reverse=True) 

	def __len__(self): 
		return len(self.word_indexes) - self.args 

	def __getitem__(self, index): 
		return ( 
			torch.tensor(self.word_indexes[index:index + self.args]), 
			torch.tensor(self.word_indexes[index + 1:index + self.args+ 1]) 
		) 
	
class LSTMModel(nn.Module): 
	def __init__(self, dataset): 
		super(LSTMModel, self).__init__() 
		self.lstm_size = 128
		self.embedding_dim = 128
		self.num_layers = 3

		n_vocab = len(dataset.unique_words) 
		self.embedding = nn.Embedding( 
			num_embeddings=n_vocab, 
			embedding_dim=self.embedding_dim, 
		) 
		self.lstm = nn.LSTM( 
			input_size=self.embedding_dim, 
			hidden_size=self.lstm_size, 
			num_layers=self.num_layers, 
			dropout=0.2, 
		) 
		self.fc = nn.Linear(self.lstm_size, n_vocab) 

	def forward(self, x, prev_state): 
		embed = self.embedding(x) 
		output, state = self.lstm(embed, prev_state) 
		logits = self.fc(output) 

		return logits, state 

	def init_state(self, sequence_length): 
		return ( 
			torch.zeros(self.num_layers, 
						sequence_length, self.lstm_size), 
			torch.zeros(self.num_layers,
						sequence_length, self.lstm_size) 
		) 

from torch.utils.data import DataLoader, random_split 

# Hyperparameters 
sequence_length = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Create the dataset 
dataset = TextDataset(sequence_length) 

# Split the dataset into training and validation sets 
train_size = int(0.8 * len(dataset)) 
val_size = len(dataset) - train_size 
train_dataset, val_dataset = random_split(dataset, 
								[train_size, val_size]) 

# Create data loaders 
train_loader = DataLoader(train_dataset, 
					batch_size=batch_size, shuffle=True) 
val_loader = DataLoader(val_dataset, 
						batch_size=batch_size) 

# Create the model 
model = LSTMModel(dataset) 

# Define the loss function and optimizer 
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(),
							lr=learning_rate) 

# Training loop 
for epoch in range(num_epochs): 
	model.train() 
	total_loss = 0.0

	for batch in train_loader: 
		inputs, targets = batch 

		optimizer.zero_grad() 

		hidden = model.init_state(sequence_length) 
		outputs, _ = model(inputs, hidden) 

		loss = criterion(outputs.view(-1, 
					len(dataset.unique_words)),
						targets.view(-1)) 
		loss.backward() 

		optimizer.step() 

		total_loss += loss.item() 

	# Calculate average loss for the epoch 
	average_loss = total_loss / len(train_loader) 

	# Print the epoch and average loss 
	print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}") 

	# Validation loop 
	model.eval() 
	val_loss = 0.0

	with torch.no_grad(): 
		for batch in val_loader: 
			inputs, targets = batch 

			hidden = model.init_state(sequence_length) 
			outputs, _ = model(inputs, hidden) 

			loss = criterion(outputs.view(-1, 
							len(dataset.unique_words)),
							targets.view(-1)) 
			val_loss += loss.item() 

	# Calculate average validation loss for the epoch 
	average_val_loss = val_loss / len(val_loader) 

	# Print the epoch and average validation loss 
	# print(f"Epoch[{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss: .4f}") 
	
    # Input a sentence 
input_sentence = "he is your "

# Preprocess the input sentence 
input_indexes = [dataset.word_to_index[word] for
				word in input_sentence.split()] 
input_tensor = torch.tensor(input_indexes,
							dtype=torch.long).unsqueeze(0) 

# Generate the next word 
model.eval() 
hidden = model.init_state(len(input_indexes)) 
outputs, _ = model(input_tensor, hidden) 
predicted_index = torch.argmax(outputs[0, -1, :]).item() 
predicted_word = dataset.index_to_word[predicted_index] 

# Print the predicted word 
print("Input Sentence:", input_sentence) 
print("Predicted Next Word:", predicted_word) 

