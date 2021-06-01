from django.shortcuts import render,redirect
from django.http import HttpResponse
from accounts.models import *
from django.contrib import messages
# Create your views here.
logi = False
logou = True

def home(request):
	return render(request, 'home.html', {'login':logi, 'logout':logou})

def logout(request):
	global logi, logou
	logi = False
	logou = True
	return render(request, 'home.html', {'login':logi, 'logout':logou})

def login(request):
	global logi, logou
	if request.method=="POST":
		username = request.POST['username']
		password = request.POST['password']
		if Users.objects.filter(username=username).exists():
			u = Users.objects.get(username=username)
			p = u.password
			if p==password:
				logi = True
				logou = False
				return redirect('home')
			else:
				return render(request, 'login1.html', {'login':logi, 'logout':logou, 'login_check':True})
		else:
			return render(request, 'login1.html', {'login':logi, 'logout':logou, 'login_check':True})
	else:
		return render(request, 'login1.html', {'login':logi, 'logout':logou, 'login_check':False})

def register(request):
	global logi, logou
	if request.method=="POST":
		first_name = request.POST['first_name']
		last_name = request.POST['last_name']
		username = request.POST['username']
		password1 = request.POST['password1']
		password2 = request.POST['password2']
		email = request.POST['email']
		if Users.objects.filter(username=username).exists():
			return render(request, 'register1.html', {'username_check':True, 'login':logi, 'logout':logou})
		if Users.objects.filter(email=email).exists():
			return render(request, 'register1.html', {'email_check':True, 'login':logi, 'logout':logou})
		if password1!=password2:
			return render(request,'register1.html', {'password_check':True, 'login':logi, 'logout':logou})
		accounts_users = Users(first_name=first_name, last_name=last_name, username=username, password=password1, email=email)
		accounts_users.save()
		return redirect('home')
	else:
		return render(request, 'register1.html', {'login':logi, 'logout':logou})