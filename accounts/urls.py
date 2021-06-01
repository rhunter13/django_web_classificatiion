from django.urls import *
from accounts import views

urlpatterns = [
	path('', views.home, name='home'),
	path('home', views.home, name='home'),
	path('login', views.login, name='login'),
	path('logout', views.logout, name='logout'),
	path('register', views.register, name='register'),
]