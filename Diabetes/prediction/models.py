from django.db import models
import json
# Create your models here.
from django.contrib.auth.models import User
from django.db import models


class UserProfile(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, null=True, blank=True)
    profile_photo = models.ImageField(upload_to='profile_photos/', null=True, blank=True)

    def __str__(self):
        return self.user.username


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateTimeField(auto_now_add=True)
    result = models.CharField(max_length=10)
    probability = models.FloatField()

    def __str__(self):
        return f'{self.user.username} - {self.result}'


from django.db import models
from django.contrib.auth.models import User

class Dataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset_file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s dataset uploaded on {self.uploaded_at}"

class AnalysisResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    column_names = models.TextField()
    head = models.TextField()
    dimensions = models.CharField(max_length=100)
    describe = models.TextField()
    missing_values = models.TextField()
    missing_values_after_handling = models.TextField()
    dataset_id = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis for {self.user.username} on {self.created_at}"

class visualizations(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    visualization_type = models.CharField(max_length=255)
    image = models.ImageField(upload_to='visualizations/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Visualization for {self.user.username} created on {self.created_at}"