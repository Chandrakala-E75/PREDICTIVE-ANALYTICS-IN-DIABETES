from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm, UserProfileForm, PredictionForm
from .models import UserProfile, Prediction, AnalysisResult, visualizations
import joblib
import io
import base64
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from django.contrib import messages
from .forms import DatasetUploadForm
from .models import Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from django.shortcuts import render
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

from io import BytesIO
from django.core.files.base import ContentFile

def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            UserProfile.objects.create(
                user=user,
                #gender=form.cleaned_data['gender']
            )

            return redirect('login')
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('profile')
        else:
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    return render(request, 'login.html')


@login_required
def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.user = request.user
            dataset.save()

            # Perform analysis
            df = pd.read_csv(dataset.dataset_file.path)
            column_names = ','.join(df.columns.tolist())
            head = df.head(10).to_html()
            dimensions = df.shape
            describe = df.describe().to_html()

            # Convert Series to DataFrame before using to_html
            missing_values = df.isnull().sum().reset_index().to_html(index=False, header=["Column", "Missing Values"])

            df_copy = df.copy(deep=True)
            df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[
                ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
            missing_values_after_handling = df_copy.isnull().sum().reset_index().to_html(index=False, header=["Column",
                                                                                                              "Missing Values After Handling"])

            # Replace previous analysis results
            AnalysisResult.objects.filter(user=request.user).delete()

            # Save new analysis results to the database
            AnalysisResult.objects.create(
                user=request.user,
                column_names=column_names,
                head=head,
                dimensions=f"{dimensions[0]} rows, {dimensions[1]} columns",
                describe=describe,
                missing_values=missing_values,
                missing_values_after_handling=missing_values_after_handling,
                dataset_id=dataset.id
            )
            request.session['dataset_id'] = dataset.id
            request.session['uploaded_file_path'] = dataset.dataset_file.path
            return redirect('analysis_result')
    else:
        form = DatasetUploadForm()

    return render(request, 'upload_dataset.html', {'form': form})


@login_required
def analysis_result(request):
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        analysis_results = AnalysisResult.objects.filter(user=request.user).order_by('-created_at').first()
        if not analysis_results:
            return redirect('upload_dataset')
        dataset_id = analysis_results.dataset_id
        request.session['dataset_id'] = dataset_id

    analysis_results = AnalysisResult.objects.filter(user=request.user, dataset_id=dataset_id).order_by(
        '-created_at').first()
    column_names = analysis_results.column_names.split(',')

    context = {
        'column_names': column_names,
        'head': analysis_results.head,
        'dimensions': analysis_results.dimensions,
        'describe': analysis_results.describe,
        'missing_values': analysis_results.missing_values,
        'missing_values_after_handling': analysis_results.missing_values_after_handling,
        'dataset_id': analysis_results.dataset_id,
    }
    return render(request, 'analysis_result.html', context)


@login_required
def visualizations(request):
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        return redirect('upload_dataset')

    analysis_results = AnalysisResult.objects.filter(user=request.user, dataset_id=dataset_id).first()
    if not analysis_results:
        return redirect('upload_dataset')

    images_dir = os.path.join('static', 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    dataset = Dataset.objects.filter(user=request.user, id=dataset_id).first()
    if not dataset:
        return redirect('upload_dataset')

    df = pd.read_csv(dataset.dataset_file.path)
    df_copy = df.copy(deep=True)
    df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

    # Data Distribution Before Imputing Missing Values
    plt.figure(figsize=(20, 20))
    df.hist(ax=plt.gca())
    plt.savefig(os.path.join(images_dir, 'data_distribution_before.png'))
    plt.close()

    # Imputing Missing Values
    df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
    df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
    df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
    df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
    df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

    # Data Distribution After Imputing Missing Values
    plt.figure(figsize=(20, 20))
    df_copy.hist(ax=plt.gca())
    plt.savefig(os.path.join(images_dir, 'data_distribution_after.png'))
    plt.close()

    # Outcome Prediction Distribution
    color_wheel = {1: "#0392cf", 2: "#7bc043"}
    colors = df["Outcome"].map(lambda x: color_wheel.get(x + 1))
    df.Outcome.value_counts().plot(kind="bar", color=colors)
    plt.savefig(os.path.join(images_dir, 'outcome_distribution.png'))
    plt.close()

    # Distribution and Outliers of Insulin
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    sns.distplot(df['Insulin'])
    plt.subplot(122)
    df['Insulin'].plot.box()
    plt.savefig(os.path.join(images_dir, 'insulin_distribution.png'))
    plt.close()

    # Correlation between all the features
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
    plt.savefig(os.path.join(images_dir, 'correlation_matrix.png'))
    plt.close()

    context = {
        'data_distribution_before': os.path.join('static/images', 'data_distribution_before.png'),
        'data_distribution_after': os.path.join('static/images', 'data_distribution_after.png'),
        'outcome_distribution': os.path.join('static/images', 'outcome_distribution.png'),
        'insulin_distribution': os.path.join('static/images', 'insulin_distribution.png'),
        'correlation_matrix': os.path.join('static/images', 'correlation_matrix.png'),
    }

    return render(request, 'visualizations.html', context)


@login_required
def prediction_results(request):
    file_path = request.session.get('uploaded_file_path')
    if not file_path:
        return render(request, 'error.html', {'error': "No file uploaded."})

    try:
        print(f"Processing file: {file_path}")

        accuracies = get_model_accuracies(file_path)

        combined_chart = create_combined_accuracy_chart(accuracies)

        context = {
            'accuracies': accuracies,
            'combined_chart': combined_chart
        }

        return render(request, 'prediction_results.html', context)
    except Exception as e:
        print(f"Error in prediction_results: {e}")
        return render(request, 'error.html', {'error': str(e)})


def get_model_accuracies(file_path):
    knn = joblib.load('prediction/Models1/knn_model.pkl')
    tree = joblib.load('prediction/Models1/tree_model.pkl')
    mlp = joblib.load('prediction/Models1/mlp_model.pkl')
    scaler = joblib.load('prediction/Models1/scaler.pkl')

    diabetes = pd.read_csv(file_path)
    X = diabetes.drop('Outcome', axis=1)
    y = diabetes['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_train_accuracy = knn.score(X_train, y_train)
    knn_test_accuracy = knn.score(X_test, y_test)

    tree_train_accuracy = tree.score(X_train, y_train)
    tree_test_accuracy = tree.score(X_test, y_test)

    mlp_train_accuracy = mlp.score(X_train_scaled, y_train)
    mlp_test_accuracy = mlp.score(X_test_scaled, y_test)

    accuracies = {
        'K-Nearest Neighbors': {
            'training_accuracy': knn_train_accuracy * 100,
            'testing_accuracy': knn_test_accuracy * 100
        },
        'Decision Tree': {
            'training_accuracy': tree_train_accuracy * 100,
            'testing_accuracy': tree_test_accuracy * 100
        },
        'Deep Learning (MLP)': {
            'training_accuracy': mlp_train_accuracy * 100,
            'testing_accuracy': mlp_test_accuracy * 100
        }
    }

    return accuracies


def create_combined_accuracy_chart(accuracies):
    models = list(accuracies.keys())
    train_accuracies = [accuracies[model]['training_accuracy'] for model in models]
    test_accuracies = [accuracies[model]['testing_accuracy'] for model in models]

    x = np.arange(len(models))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, train_accuracies, width, label='Training Accuracy', color='blue')
    bars2 = ax.bar(x + width / 2, test_accuracies, width, label='Testing Accuracy', color='orange')

    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training vs. Testing Accuracy by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the plot

    # Adding accuracy values on top of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()  # Adjust layout to make room for the legend

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return chart

@login_required
def profile(request):
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully')
            return redirect('profile')
    else:
        form = UserProfileForm(instance=profile)

    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'profile.html', {'form': form, 'predictions': predictions, 'profile': profile})






@login_required
def predict(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Load the trained model and scaler
            model = joblib.load('prediction/models/trained_model.pkl')
            scaler = joblib.load('prediction/models/scaler.pkl')

            # Extract form data
            data = [
                form.cleaned_data['pregnancies'],
                form.cleaned_data['glucose'],
                form.cleaned_data['blood_pressure'],
                form.cleaned_data['skin_thickness'],
                form.cleaned_data['insulin'],
                form.cleaned_data['bmi'],
                form.cleaned_data['diabetes_pedigree_function'],
                form.cleaned_data['age']
            ]

            # Scale and predict
            data = scaler.transform([data])
            result = model.predict(data)[0]
            probability = model.predict_proba(data)[0][result]

            # Save prediction
            Prediction.objects.create(
                user=request.user,
                result='Positive' if result == 1 else 'Negative',
                probability=probability * 100
            )

            return redirect('result', result='Positive' if result == 1 else 'Negative', probability=str(probability * 100))
    else:
        form = PredictionForm()
    return render(request, 'predict.html', {'form': form})

@login_required
def result(request, result, probability):
    probability = float(probability)
    return render(request, 'result.html', {'result': result, 'probability': probability})

def user_logout(request):
    logout(request)
    return redirect('home')
