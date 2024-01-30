import pickle
from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import AuthenticationForm

from DearDiary.autocomplete_ai import AutocompleteAI
from .forms import UserCreationForm
from django.db.models import Count, Avg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

import numpy as np
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect('home')
        else:
            print(form.errors)
    else:
        form = UserCreationForm()
    
    return render(request, 'registration/register.html', {'form': form})

def login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('home')
        else:
            print(form.errors)
            # Handle authentication errors
            if 'username' in form.errors and 'password' in form.errors:
                # Both username and password are invalid
                form.add_error(None, 'Invalid username and password')
            elif 'username' in form.errors:
                # Only the username is invalid
                form.add_error(None, 'Account not found')
    else:
        form = AuthenticationForm()

    return render(request, 'registration/login.html', {'form': form})

def logout(request):
    auth_logout(request)
    return redirect('home')

# def profile(request):
#     return render(request,'home.html')
# def get_autocomplete_suggestions(request):
#     prefix = request.GET.get('prefix', '')
    
#     autocomplete_ai = AutocompleteAI()  # Initialize or get an instance of AutocompleteAI

#     file_path = 'indonesian_words.txt'

#     # Read the content of the text file and store each line as a string in a list
#     with open(file_path, 'r') as txt_file:
#         lines = txt_file.readlines()

#     # Strip leading and trailing whitespaces from each line
#     lines = [line.strip() for line in lines]
    
#     # Assuming you have a dataset of Indonesian words
#     indonesian_words = lines 

#     for word in indonesian_words:
#         autocomplete_ai.insert(word)

#     # Split the sentence into words and get the last word
#     words = prefix.split()
#     if words:
#         last_word = words[-1]
#         suggestions = autocomplete_ai.search(last_word)
#         print(f"Autocomplete suggestions for '{last_word}': {suggestions}")

#     data = {
#         'suggestions': suggestions,
#     }
    
#     print(f"Prefix: {prefix}, Suggestions: {suggestions}")
#     return JsonResponse(data)

def generate_text(seed_text, next_words, model, max_sequence_len):

    # Load the tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

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

def get_lstm_suggestions(request):

    # Load the trained model during Django app initialization
    model = load_model('lstm_autocomplete.keras')

    prefix = request.GET.get('prefix', '')

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

    seed_text = prefix
    next_words = 5
    generated_text = generate_text(seed_text, next_words, model, max_sequence_len)
    return JsonResponse(generated_text, safe=False)

    
def home(request):
    user = request.user
    return render(request,'home.html')

def input(request):
    return render(request,'input.html')

@login_required
def watch(request, anime_id):
    return render(request,'watch.html')