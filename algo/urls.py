from django.urls import *
from algo import views

urlpatterns = [
	path('',views.regression, name='ccc'),
	path('regression', views.regression, name='regression'),
	path('classification', views.classification, name='class'),
	path('upload', views.upload, name='upload'),
	path('classificationtask', views.classificationtask, name='classificationtask'),
	path('regressiontask', views.regressiontask, name='regressiontask'),
	path('task', views.task, name="task"),
	path('classificationplots', views.classificationplots, name='classificationplots'),
	path('regressionplots', views.regressionplots, name='regressionplots'),
]