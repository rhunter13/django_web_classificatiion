from django.db import models	

# Create your models here.
class Data_sets(models.Model):
	train_data = models.FileField(upload_to='files')
