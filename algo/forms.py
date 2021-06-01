from django import forms
from algo.models import Data_sets

class Upload_form(forms.ModelForm):
	class Meta:
		model = Data_sets
		fields = ('train_data',)