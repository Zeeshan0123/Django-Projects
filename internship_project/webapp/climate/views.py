from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
import os
from .utils import predict_temperature, train_model

# Create your views here.

def index(request):
    return HttpResponse("Hi,  Welcome to the Global temperature prediction app.")


@csrf_exempt
def train(request):
    if request.method == 'POST':
        # csv_file = request.FILES['file']
        # file_name = default_storage.save(csv_file.name, csv_file)
        # file_path = os.path.join(default_storage.location, file_name)
        file_path = "updated.csv"
        train_model(file_path)
        return JsonResponse({"message": "Model trained and saved successfully."})
    return JsonResponse({"error": "Only POST method is allowed."}, status=400)



@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            features = [
                float(data['LandAverageTemperature']),
                float(data['LandMaxTemperature']),
                float(data['LandMinTemperature'])
            ]
            print("Features:", features)
            
            MODEL_PATH = "model.pkl"
            # load the model
            if not os.path.exists(MODEL_PATH):
                return JsonResponse({"error":"Model not trained yet. Please train the model first."},status=400)
            
            knn_model = joblib.load(MODEL_PATH)
            
            # predict using the model 
            prediction = knn_model.predict([features])
            return JsonResponse({"predicted_temperature = ":round(prediction[0],3)})
        except KeyError as e:
            return JsonResponse({"error":f"Missing key: {e}"},status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Only POST method is allowed."}, status=400)



    



