from django.contrib import admin
from django.urls import path
from vsm_working import views

urlpatterns = [
    path("",views.index, name='vsm_start')
]
