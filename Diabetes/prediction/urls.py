from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    #path('analysis/', views.analysis, name='analysis'),
    path('analysis_result/', views.analysis_result, name='analysis_result'),
    path('visualizations/', views.visualizations, name='visualizations'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('register/', views.register, name='register'),
    path('profile/', views.profile, name='profile'),
    path('predict/', views.predict, name='predict'),
    path('result/<str:result>/<str:probability>/', views.result, name='result'),
    path('prediction-results/', views.prediction_results, name='prediction_results'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
