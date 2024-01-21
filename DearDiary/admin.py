from django.contrib import admin
from .models import Diary

class DiaryAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'created_at')
    search_fields = ('title', 'user__username')  
    
admin.site.register(Diary, DiaryAdmin)