from django import forms
from django.contrib.auth.models import User
from .models import UserProfile


from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
import re
from django import forms
from .models import Dataset

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['dataset_file']


class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput, min_length=8)
    confirm_password = forms.CharField(widget=forms.PasswordInput, label='Confirm Password')
    email = forms.EmailField()
   # gender = forms.ChoiceField(choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')])

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'confirm_password']

    def clean_password(self):
        password = self.cleaned_data.get('password')
        if not re.findall('\d', password):
            raise ValidationError("The password must contain at least one digit.")
        if not re.findall('[A-Za-z]', password):
            raise ValidationError("The password must contain at least one letter.")
        if not re.findall('[^A-Za-z0-9]', password):
            raise ValidationError("The password must contain at least one special character.")
        return password

    def clean_confirm_password(self):
        password = self.cleaned_data.get('password')
        confirm_password = self.cleaned_data.get('confirm_password')
        if password and confirm_password and password != confirm_password:
            raise ValidationError("Passwords don't match.")
        return confirm_password

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError("Email already exists.")
        return email



class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['age', 'gender', 'profile_photo']
        widgets = {
            'age': forms.NumberInput(attrs={'class': 'form-control'}),
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'profile_photo': forms.FileInput(attrs={'class': 'form-control'}),
        }


class PredictionForm(forms.Form):
    # Add fields for prediction parameters
    pregnancies = forms.IntegerField()
    glucose = forms.IntegerField()
    blood_pressure = forms.IntegerField()
    skin_thickness = forms.IntegerField()
    insulin = forms.IntegerField()
    bmi = forms.FloatField()
    diabetes_pedigree_function = forms.FloatField()
    age = forms.IntegerField()
