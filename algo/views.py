from django.shortcuts import render,redirect
from django.http import HttpResponse
from algo.models import Data_sets
from django.contrib import messages
from machine_learning.settings import MEDIA_ROOT
import accounts.views as v

def delete_files():
	import os
	import glob

	files = glob.glob('/home/venkat/Desktop/machine_learning/media/files/*')
	for f in files:
		os.remove(f)

def classification(request):
	if request.method=="POST":
		delete_files()
		t = request.FILES['train_data']
		k = str(t)
		pos = k.find('.')
		extension = k[pos+1:]
		if v.logi==False:
			return render(request,'classification.html', {'loggedin':True, 'login':v.logi, 'logout':v.logou})
		if extension=='csv' or extension=='xlsx':
			Data_sets.objects.all().delete()
			algo_data_sets = Data_sets(train_data=t)
			algo_data_sets.save()
			return redirect("classificationplots")
		else:
			return render(request,'classification.html', {'check':True, 'login':v.logi, 'logout':v.logou})

	return render(request, 'classification.html', {'check':False, 'login':v.logi, 'logout':v.logou})

def regression(request):
	if request.method=="POST":
		t = request.FILES['train_data']
		k = str(t)
		pos = k.find('.')
		extension = k[pos+1:]
		if v.logi==False:
			return render(request,'regression.html', {'loggedin':True, 'login':v.logi, 'logout':v.logou})
		if extension=='csv' or extension=='xlsx':
			algo_data_sets = Data_sets(train_data=t)
			algo_data_sets.save()
			return redirect("regressionplots")
		else:
			return render(request,'regression.html', {'check':True, 'login':v.logi, 'logout':v.logou})

	return render(request, 'regression.html', {'check':False, 'login':v.logi, 'logout':v.logou})

class plots():
	name:str
	url:str


def classificationplots(request):
	import pandas as pd
	from pandas_profiling import ProfileReport
	dat = Data_sets.objects.all()
	required_data = dat[len(dat)-1]
	path = MEDIA_ROOT+'/'+str(required_data.train_data)
	k = str(required_data.train_data)
	pos = k.find('.')
	extension = k[pos+1:]
	if extension=='csv':
		data = pd.read_csv(path)
	else:
		data = pd.read_excel(path)
	
	profile = ProfileReport(data, title='Pandas Profiling Report', explorative=True)
	base = a = '<!DOCTYPE html><html lang="en">    <head>        <meta charset="UTF-8">        <title>Machine Learning</title>        <style>        	.top{                background-color: black;                text-align: center;                text-justify: auto;                font-size: xx-large;                opacity:1;            }            a{                color: #f0ffff;                text-decoration: none;            }        </style>    </head>    <body>    	<div class="top">    <a href="classificationtask">Perform Classification</a>    	</div>    </body></html>'
	html = profile.to_html()
	#profile.to_file('/home/venkat/Desktop/machine_learning/templates/'+k[6:pos]+'.html')
	#base_path = '/home/venkat/Desktop/machine_learning/templates/'
	#file = base_path+'modified'+ str(k[6:pos]+'.html')
	#file1 = open(base_path+'base.html', 'r')
	#file2 = open(base_path+k[6:pos]+'.html', 'r')
	#file3 = open(base_path+'modified'+k[6:pos]+'.html', 'w')
	#file3.write(file1.read())
	#file3.write('\n')
	#file3.write(file2.read())
	return HttpResponse(base+html)	

def regressionplots(request):
	import pandas as pd
	from pandas_profiling import ProfileReport
	dat = Data_sets.objects.all()
	required_data = dat[len(dat)-1]
	path = MEDIA_ROOT+'/'+str(required_data.train_data)
	k = str(required_data.train_data)
	pos = k.find('.')
	extension = k[pos+1:]
	if extension=='csv':
		data = pd.read_csv(path)
	else:
		data = pd.read_excel(path)
	
	profile = ProfileReport(data, title='Pandas Profiling Report')
	profile.to_file('/home/venkat/Desktop/machine_learning/templates/'+k[6:pos]+'.html')
	base_path = '/home/venkat/Desktop/machine_learning/templates/'
	file = base_path+'modified'+ str(k[6:pos]+'.html')
	file1 = open(base_path+'base1.html', 'r')
	file2 = open(base_path+k[6:pos]+'.html', 'r')
	file3 = open(base_path+'modified'+k[6:pos]+'.html', 'w')
	file3.write(file1.read())
	file3.write('\n')
	file3.write(file2.read())
	return render(request, base_path+'modified'+k[6:pos]+'.html')

def upload(request):
	return HttpResponse("upload")

class Values():
	index:int
	actual:str
	predicted:str

def task(request):
	return HttpResponse("hello")

def ravel(y_true,y_pred, p, n):
    tp=0
    fp=0
    tn=0
    fn=0 
    for i in range(len(y_true)):
        if y_true[i]==n and y_pred[i]==n:
            tn+=1
        elif y_true[i]==n and y_pred[i]==p:
            fp+=1
        elif y_true[i]==p and y_pred[i]==n:
            fn+=1
        elif y_true[i]==p and y_pred[i]==p:
            tp+=1
    return (tp,tn,fn,fp)

def remove_punctuation(s):
	import string
	return s.translate(str.maketrans('','', string.punctuation))

def classificationtask(request):
	import pandas as pd
	import sklearn
	from sklearn import ensemble, model_selection, tree, linear_model, svm, metrics, neighbors
	from sklearn import preprocessing, feature_extraction, metrics
	import numpy as np
	import scipy
	from scipy.sparse import csr_matrix
	dat = Data_sets.objects.all()
	required_data = dat[len(dat)-1]
	path = MEDIA_ROOT+'/'+str(required_data.train_data)
	k = str(required_data.train_data)
	pos = k.find('.')
	extension = k[pos+1:]
	if extension=='csv':
		data = pd.read_csv(path)
	else:
		data = pd.read_excel(path)
	features = list(data.columns)

	if request.method=="POST":
		model = request.POST["model"]
		inp=list()
		for i in features:
			try:
				a = request.POST[i]
				inp.append(a)
			except:
				pass
		target = request.POST['target']
		if len(inp)==0:
			return render(request,'classificationtask.html', {'feature_check':True, 'target_check':False, 'login':v.logi, 'logout':v.logou,'features':features})
		if data[target].dtype == np.dtype('float64'):
			return render(request,'classificationtask.html', {'feature_check':False, 'target_check':True, 'login':v.logi, 'logout':v.logou,'features':features}) 
		l = inp+[target]
		new_data = data[l]
		miss = dict()
		for i in list(new_data.columns):
			if new_data[i].dtype == np.dtype('O'):
				miss[i] = new_data[i].mode()[0]
			else:
				miss[i] = new_data[i].mean()
		new_data = new_data.fillna(value=miss)
		#train_valid,test = sklearn.model_selection.train_test_split(data,train_size=0.8, random_state=0)
		predictor = list()
		for i in inp:
			if new_data[i].dtype==np.dtype('O'):
				string = data[i].str.contains(' ')
				if True in string:
					ve = sklearn.feature_extraction.text.CountVectorizer()
					predictor.append(ve.fit_transform(new_data[i]))
				else:
					le = sklearn.preprocessing.LabelEncoder()
					predictor.append(le.fit_transform(new_data[i].to_numpy().reshape(-1,1)))
			else:
				predictor.append(new_data[i].to_numpy().reshape(-1,1))
		x = csr_matrix(predictor[0])
		if len(predictor)>1:
			for i in range(1, len(predictor)):
				temp = csr_matrix(predictor[i])
				x = scipy.sparse.hstack([x,temp])
		y = new_data[target].to_numpy()
		x = sklearn.preprocessing.normalize(x, axis=0)	
		x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,train_size=0.8, random_state=0)
		
		if model=="decision_trees":
			m = sklearn.tree.DecisionTreeClassifier()
			parameters = {'max_depth':[5,10,15,20]}
			algo = sklearn.model_selection.GridSearchCV(m, parameters)
			algo.fit(x_train,y_train)
	
		elif model=="knn":
			m = sklearn.neighbors.KNeighborsClassifier()
			parameters = {'n_neighbors':[5,10,15,20]}
			algo = sklearn.model_selection.GridSearchCV(m, parameters)			 
			algo.fit(x_train,y_train)
		elif model=='random_forest':
			algo = sklearn.ensemble.RandomForestClassifier()
			algo.fit(x_train,y_train)
		elif model=='bagging_knn':
			base = sklearn.neighbors.KNeighborsClassifier()
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)			 
			algo.fit(x_train,y_train)
		elif model=='bagging_decision':
			base = sklearn.tree.DecisionTreeClassifier()
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)					 
			algo.fit(x_train,y_train)
		elif model=='adaboost_decision':
			base = sklearn.tree.DecisionTreeClassifier()
			try:
				algo = sklearn.ensemble.AdaBoostClassifier(base_estimator=base)
			except:
				algo = sklearn.ensemble.AdaBoostClassifier(base_estimator=base, algorithm='SAMME')			
			algo.fit(x_train,y_train)
		elif model=='bagging_log':
			base = sklearn.linear_model.LogisticRegression(max_iter=1000)
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)		 
			algo.fit(x_train,y_train)
		elif model=='adaboost_log':
			base = sklearn.linear_model.LogisticRegression(max_iter=1000)
			try:
				algo = sklearn.ensemble.AdaBoostClassifier(base_estimator=base)
			except:
				algo = sklearn.ensemble.AdaBoostClassifier(base_estimator=base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		elif model=="logistic_regression":
			algo = sklearn.linear_model.LogisticRegression(max_iter=10000)
			algo.fit(x_train,y_train)
		elif model=="svm":
			algo = sklearn.svm.SVC()
			algo.fit(x_train,y_train)
		elif model=='bagging_svm':
			base = sklearn.svm.SVC()
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)
			algo.fit(x_train,y_train)
		elif model=='adaboost_svm':
			base = sklearn.svm.SVC()
			algo = sklearn.ensemble.AdaBoostClassifier(base_estimator=base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		elif model=="sgd_hinge":
			algo = sklearn.linear_model.SGDClassifier(loss='hinge')
			algo.fit(x_train,y_train)
		elif model=='bagging_sgd(hinge)':
			base = sklearn.linear_model.SGDClassifier(loss='hinge')
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)
			algo.fit(x_train,y_train)
		elif model=='adaboost_sgd(hinge)':
			base = sklearn.linear_model.SGDClassifier(loss='hinge')
			algo = sklearn.ensemble.AdaBoostClassifier(base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		elif model=="sgd_log":
			algo = sklearn.linear_model.SGDClassifier(loss='log')
			algo.fit(x_train,y_train)
		elif model=='bagging_sgd(log)':
			base = sklearn.linear_model.SGDClassifier(loss='log')
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)
			algo.fit(x_train,y_train)
		elif model=='adaboost_sgd(log)':
			base = sklearn.linear_model.SGDClassifier(loss='log')
			algo = sklearn.ensemble.AdaBoostClassifier(base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		elif model=="sgd_modifiedhuber":
			algo = sklearn.linear_model.SGDClassifier(loss='modified_huber')
			algo.fit(x_train,y_train)
		elif model=='bagging_sgd(modified_huber)':
			base = sklearn.linear_model.SGDClassifier(loss='modified_huber')
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)
			algo.fit(x_train,y_train)	
		elif model=='adaboost_sgd(modified_huber)':
			base = sklearn.linear_model.SGDClassifier(loss='modified_huber')
			algo = sklearn.ensemble.AdaBoostClassifier(base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		elif model=="sgd_squaredhinge":
			algo = sklearn.linear_model.SGDClassifier(loss='squared_hinge')
			algo.fit(x_train,y_train)
		elif model=='bagging_sgd(squared_hinge)':
			base = sklearn.linear_model.SGDClassifier(loss='squared_hinge')
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)
			algo.fit(x_train,y_train)
		elif model=='adaboost_sgd(squared_hinge)':
			base = sklearn.linear_model.SGDClassifier(loss='squared_hinge')
			algo = sklearn.ensemble.AdaBoostClassifier(base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		elif model=="sgd_perceptron":
			algo = sklearn.linear_model.SGDClassifier(loss='perceptron')
			algo.fit(x_train,y_train)
		elif model=='bagging_sgd(perceptron)':
			base = sklearn.linear_model.SGDClassifier(loss='perceptron')
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)
			algo.fit(x_train,y_train)
		elif model=='adaboost_sgd(perceptron)':
			base = sklearn.linear_model.SGDClassifier(loss='perceptron')
			algo = sklearn.ensemble.AdaBoostClassifier(base_estimator=base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		elif model=="sgd_squaredloss":
			algo = sklearn.linear_model.SGDClassifier(loss='squared_loss')
			algo.fit(x_train,y_train)
		elif model=='bagging_sgd(squaredloss)':
			base = sklearn.linear_model.SGDClassifier(loss='squared_loss')
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)
			algo.fit(x_train,y_train)
		elif model=='adaboost_sgd(squaredloss)':
			base = sklearn.linear_model.SGDClassifier(loss='squared_loss')
			algo = sklearn.ensemble.AdaBoostClassifier(base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		elif model=="sgd_huber":
			algo = sklearn.linear_model.SGDClassifier(loss='huber')
			algo.fit(x_train,y_train)
		elif model=='bagging_sgd(huber)':
			base = sklearn.linear_model.SGDClassifier(loss='huber')
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)
			algo.fit(x_train,y_train)
		elif model=='adaboost_sgd(huber)':
			base = sklearn.linear_model.SGDClassifier(loss='huber')
			algo = sklearn.ensemble.AdaBoostClassifier(base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		elif model=="sgd_epsiloninsensitive":
			algo = sklearn.linear_model.SGDClassifier(loss='epsilon_insensitive')
			algo.fit(x_train,y_train)
		elif model=='bagging_sgd(epsiloninsensitive)':
			base = sklearn.linear_model.SGDClassifier(loss='epsilon_insensitive')
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)
			algo.fit(x_train,y_train)
		elif model=='adaboost_sgd(epsiloninsensitive)':
			base = sklearn.linear_model.SGDClassifier(loss='epsilon_insensitive')
			algo = sklearn.ensemble.AdaBoostClassifier(base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		elif model=="sgd_squaredepsiloninsensitive":
			algo = sklearn.linear_model.SGDClassifier(loss='squared_epsilon_insensitive')
			algo.fit(x_train,y_train)
		elif model=='bagging_sgd(squared_epsiloninsensitive':
			base = sklearn.linear_model.SGDClassifier(loss='squared_epsilon_insensitive')
			algo = sklearn.ensemble.BaggingClassifier(base_estimator=base)
			algo.fit(x_train,y_train)
		elif model=='adaboost_sgd(squared_epsiloninsensitive':
			base = sklearn.linear_model.SGDClassifier(loss='squared_epsilon_insensitive')
			algo = sklearn.ensemble.AdaBoostClassifier(base, algorithm='SAMME')
			algo.fit(x_train,y_train)
		test_true = y_test
		test_pre = algo.predict(x_test)
		output = dict()
		output['login']=v.logi
		output['logout'] = v.logou
		output['accuracy'] = algo.score(x_test,y_test)
		output['precision'] = sklearn.metrics.precision_score(test_true, test_pre, average='micro')
		output['recall'] = sklearn.metrics.recall_score(test_true, test_pre, average='micro')
		actual_class = np.empty(len(algo.classes_), dtype='U21')
		for i in range(len(actual_class)):
			actual_class[i] = 'Actual Class '+str(algo.classes_[i])
		predicted_class = np.empty(len(algo.classes_), dtype='U21')
		for i in range(len(actual_class)):
			predicted_class[i] = 'Predicted Class '+str(algo.classes_[i])
		matrix = sklearn.metrics.confusion_matrix(test_true,test_pre,labels=algo.classes_).astype('str')
		classes = actual_class.reshape(-1,1)
		matrix = np.concatenate((classes,matrix.astype('str')), axis=1)
		g = np.insert(predicted_class, 0, '', axis=0)
		g=g.reshape(1,-1)
		matrix = np.concatenate((g,matrix), axis=0)
		output['mat'] = matrix
		output['mathew'] = sklearn.metrics.matthews_corrcoef(test_true, test_pre)
		output['kappa'] = sklearn.metrics.cohen_kappa_score(test_true,test_pre)
		l = list()
		pre = algo.predict(x_test)
		ab = y_test
		for i in range(len(y_test)):
			a = Values()
			a.index = i+1
			a.actual = ab[i]
			a.predicted = pre[i]
			l.append(a)
		output['values'] = l
		return render(request, 'classification_output.html', output)
	else:
		return render(request,'classificationtask.html', {'feature_check':False, 'target_check':False,'login':v.logi, 'logout':v.logou,'features':features})

def regressiontask(request):
	import pandas as pd
	import sklearn
	from sklearn import linear_model, model_selection
	import numpy as np
	dat = Data_sets.objects.all()
	required_data = dat[len(dat)-1]
	path = MEDIA_ROOT+'/'+str(required_data.train_data)
	k = str(required_data.train_data)
	pos = k.find('.')
	extension = k[pos+1:]
	if extension=='csv':
		data = pd.read_csv(path)
	else:
		data = pd.read_excel(path)
	features = list(data.columns)
	if request.method=="POST":
		model = request.POST["model"]
		inp=list()
		for i in features:
			try:
				a = request.POST[i]
				inp.append(a)
			except:
				pass
		target = request.POST['target']
		if target in inp:
			inp.remove(target)
		train_valid,test = sklearn.model_selection.train_test_split(data,train_size=0.8, random_state=0)
		if model=="linear_regression":
			m = sklearn.linear_model.LinearRegression()
			 
			m.fit(train_valid[inp].to_numpy(), train_valid[target].to_numpy())
			p = np.dot(test[inp].to_numpy(), m.coef_)+m.intercept_
			y = test[target].to_numpy()
			error=0.0
			for i in range(len(y)):
				error += (y[i]-p[i])**2
			error /= len(y)
			pre = np.dot(data[inp].to_numpy(), m.coef_)+m.intercept_
			output = dict()
			output['error'] = error
			l = list()
			ab = data[target].to_numpy()
			for i in range(len(data)):
				a = Values()
				a.index = i+1
				a.actual = ab[i]
				a.predicted = round(pre[i],2)
				l.append(a)
			output['values'] = l
			return render(request, 'regression_output.html', output)
		elif model=="ridge_regression":
			m =sklearn.linear_model.RidgeCV(normalize=True, cv=5)
			 
			m.fit(train_valid[inp].to_numpy(), train_valid[target].to_numpy())
			p = np.dot(test[inp].to_numpy(), m.coef_)
			y = test[target].to_numpy()
			error=0.0
			for i in range(len(y)):
				error += (y[i]-p[i]-m.intercept_)**2
			error /= len(y)
			pre = np.dot(data[inp].to_numpy(), m.coef_)
			output = dict()
			output['error'] = error
			l = list()
			ab = data[target].to_numpy()
			for i in range(len(data)):
				a = Values()
				a.index = i+1
				a.actual = ab[i]
				a.predicted = round(pre[i]+m.intercept_,2)
				l.append(a)
			output['values'] = l
			return render(request, 'regression_output.html', output)
		elif model=="lasso_regression":
			m =sklearn.linear_model.LassoCV(normalize=True, cv=5)
			 
			m.fit(train_valid[inp].to_numpy(), train_valid[target].to_numpy())
			p = np.dot(test[inp].to_numpy(), m.coef_)
			y = test[target].to_numpy()
			error=0.0
			for i in range(len(y)):
				error += (y[i]-p[i]-m.intercept_)**2
			error /= len(y)
			pre = np.dot(data[inp].to_numpy(), m.coef_)
			output = dict()
			output['error'] = error
			l = list()
			ab = data[target].to_numpy()
			for i in range(len(data)):
				a = Values()
				a.index = i+1
				a.actual = ab[i]
				a.predicted = round(pre[i]+m.intercept_,2)
				l.append(a)
			output['values'] = l
			return render(request, 'regression_output.html', output)
		elif model=='elasticnet':
			m =sklearn.linear_model.ElasticNetCV(normalize=True, cv=5)
			 
			m.fit(train_valid[inp].to_numpy(), train_valid[target].to_numpy())
			p = np.dot(test[inp].to_numpy(), m.coef_)
			y = test[target].to_numpy()
			error=0.0
			for i in range(len(y)):
				error += (y[i]-p[i]-m.intercept_)**2
			error /= len(y)
			pre = np.dot(data[inp].to_numpy(), m.coef_)
			output = dict()
			output['error'] = error
			l = list()
			ab = data[target].to_numpy()
			for i in range(len(data)):
				a = Values()
				a.index = i+1
				a.actual = ab[i]
				a.predicted = round(pre[i]+m.intercept_,2)
				l.append(a)
			output['values'] = l
			return render(request, 'regression_output.html', output)
	else:
		return render(request,'regressiontask.html', {'features':features})