from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import AuthenticationForm
from .forms import UserCreationForm
from django.db.models import Count, Avg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

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
    
def home(request):
    user = request.user

    # context = {
    #     'trending': trendingAnimes,
    #     'genreRecs': genre_recommendations,
    #     'anime_title': anime_title,
    #     'auth_user': auth_user,
    #     'watched': watched
    # }
    return render(request,'home.html')

@login_required
def watch(request, anime_id):
    return render(request,'watch.html')