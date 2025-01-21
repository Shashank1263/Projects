from django.shortcuts import render
from .models import *
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from django.contrib.auth import authenticate,login,logout
from django.http import HttpResponse

# Create your views here.
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

@csrf_exempt
def new(request):
    if request.method == "POST":
        data = request.POST
        new_employee_data = {
            'satisfaction_level': [float(data.get('satisfaction_level'))],
            'last_evaluation': [float(data.get('last_evaluation'))],
            'number_project': [int(data.get('number_project'))],
            'average_montly_hours': [int(data.get('average_montly_hours'))],
            'time_spend_company': [int(data.get('time_spend_company'))],
            'Work_accident': [int(data.get('Work_accident'))],
            'promotion_last_5years': [int(data.get('promotion_last_5years'))],
            'Department': [data.get('Department')],
            'salary': [data.get('salary')],
        }

        path = "C:\\Users\\shash\\Downloads\\Datanew\\Datanew\\employeeretention.csv"
        df = pd.read_csv(path)

        le_salary = LabelEncoder()
        le_Department = LabelEncoder()
        df['salary'] = le_salary.fit_transform(df['salary'])
        df['Department'] = le_Department.fit_transform(df['Department'])

        X = df.drop(['left'], axis=1)
        y = df['left']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = SVC(kernel='rbf', C=10, gamma='scale')
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # print(f"Accuracy: {accuracy:.2f}")
        # print("Confusion Matrix:")
        # print(conf_matrix)
        # print("Classification Report:")
        # print(classification_rep)

        new_employee_df = pd.DataFrame(new_employee_data)

        new_employee_df['salary'] = new_employee_df['salary'].apply(
            lambda x: x if x in le_salary.classes_ else le_salary.classes_[0])
        new_employee_df['Department'] = new_employee_df['Department'].apply(
            lambda x: x if x in le_Department.classes_ else le_Department.classes_[0])
        new_employee_df['salary'] = le_salary.transform(new_employee_df['salary'])
        new_employee_df['Department'] = le_Department.transform(new_employee_df['Department'])
        
        new_employee_scaled = scaler.transform(new_employee_df)

        prediction = model.predict(new_employee_scaled)

        if prediction == 1:
            result_message = "Person is likely to leave the company."
        else:
            result_message = "Person is likely to stay in the company."

        return render(request, 'new.html', context={'result': result_message})
    
    return render(request, 'new.html')

def about(request):
    return render(request,'about.html')

def index(request):
    info ="welcome to django 1st session"
    return render(request,'index.html',context={'info':info})

def employee(request):
    if(request.method=="POST"):
        data= request.POST
        firstname = data.get('textfirstname')
        lastname=data.get('textlastname')
        if('submit' in request.POST ):
            result= firstname + " " + lastname
            return render(request,'employee.html',context={'result':result})
    return render(request,'employee.html')

def calci(request):
    if(request.method=="POST"):
        data=request.POST
        num1=data.get('textfirstnumber')
        num2=data.get('textsecondnumber')
        if(len(num1)<1):
            result="Enter the first number"
            return render(request,'calci.html',context={'result': result})
        if(len(num2)<1):
            result="Enter the second number"
            return render(request,'calci.html',context={'result': result})
        if('buttonadd' in request.POST):
            result= int(num1)+int(num2)
            return render(request,'calci.html',context={'result': result})
        if('buttonsub' in request.POST):
            result= int(num1)-int(num2)
            return render(request,'calci.html',context={'result': result})
        if('buttonmul' in request.POST):
            result= int(num1)*int(num2)
            return render(request,'calci.html',context={'result': result})
        if('buttondiv' in request.POST):
            result= int(num1)/int(num2)
            return render(request,'calci.html',context={'result': result})
        
    return render(request,'calci.html')



def userlogin(request):
    if request.method=="POST":
        data=request.POST
        username = data.get('textusername')
        password=data.get('textpassword')
        user= User.objects.filter(username=username)
        if not user.exists():
            result="Invalid username"
            return render(request,'login.html',context={'result':result}) 
        user= authenticate(username=username,password=password)
        if(user is None):
            result="Invalid password"
            return render(request,'login.html',context={'result':result})
        else:
            login(request,user)
            return redirect('/home/') 

    return render(request,'login.html') 


def userlogout(request):
    logout(request)
    return redirect('/userlogin')

def Classification(request):
    if request.method=="POST" and request.FILES['myfile']:
        myfile=request.FILES['myfile']
        fs= FileSystemStorage()
        filename= fs.save("uploads//"+myfile.name,myfile)
        result= classify(myfile.name)        
        return render(request,'Classification.html',context={'result':result})

    return render(request,'Classification.html')

def classify(img_file):    
    data = []
    labels = []
    classes = 2
    cur_path = "..\\code\\"#os.getcwd() #To get current directory


    classs = { 1:"Elephant:Elephants are the largest living land animals. Three living species are currently recognised: the African bush elephant (Loxodonta africana), the African forest elephant (L. cyclotis), and the Asian elephant (Elephas maximus). They are the only surviving members of the family Elephantidae and the order Proboscidea; extinct relatives include mammoths and mastodons. Distinctive features of elephants include a long proboscis called a trunk, tusks, large ear flaps, pillar-like legs, and tough but sensitive grey skin. The trunk is prehensile, bringing food and water to the mouth and grasping objects. Tusks, which are derived from the incisor teeth, serve both as weapons and as tools for moving objects and digging. The large ear flaps assist in maintaining a constant body temperature as well as in communication. African elephants have larger ears and concave backs, whereas Asian elephants have smaller ears and convex or level backs.",
    2:"Zebra:Zebras (US: /ˈziːbrəz/, UK: /ˈzɛbrəz, ˈziː-/)[2] (subgenus Hippotigris) are African equines with distinctive black-and-white striped coats. There are three living species: Grévy's zebra (Equus grevyi), the plains zebra (E. quagga), and the mountain zebra (E. zebra). Zebras share the genus Equus with horses and asses, the three groups being the only living members of the family Equidae. Zebra stripes come in different patterns, unique to each individual. Several theories have been proposed for the function of these patterns, with most evidence supporting them as a deterrent for biting flies. Zebras inhabit eastern and southern Africa and can be found in a variety of habitats such as savannahs, grasslands, woodlands, shrublands, and mountainous areas.",
    3:"Lion:The lion (Panthera leo) is a large cat of the genus Panthera, native to Africa and India. It has a muscular, broad-chested body; a short, rounded head; round ears; and a hairy tuft at the end of its tail. It is sexually dimorphic; adult male lions are larger than females and have a prominent mane. It is a social species, forming groups called prides. A lion's pride consists of a few adult males, related females, and cubs. Groups of female lions usually hunt together, preying mostly on large ungulates. The lion is an apex and keystone predator; although some lions scavenge when opportunities occur and have been known to hunt humans, lions typically do not actively seek out and prey on humans.",
    4:"Cheetah:The cheetah (Acinonyx jubatus) is a large cat and the fastest land animal. It has a tawny to creamy white or pale buff fur that is marked with evenly spaced, solid black spots. The head is small and rounded, with a short snout and black tear-like facial streaks. It reaches 67–94 cm (26–37 in) at the shoulder, and the head-and-body length is between 1.1 and 1.5 m (3 ft 7 in and 4 ft 11 in). Adults weigh between 21 and 72 kg (46 and 159 lb). The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail."}


    
    model = load_model("C:\\Users\\shash\\OneDrive\\Desktop\\project shashi\\my_model.h5")
    print("Loaded model from disk");
    path2="uploads//"+img_file
    print(path2)
    test_image = Image.open(path2)
    test_image = test_image.resize((30, 30))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.array(test_image)
        #result = model.predict_classes(test_image)[0]	
    predict_x=model.predict(test_image)
    result=np.argmax(predict_x,axis=1)
    sign = classs[int(result) + 1]        
    print(sign) 
    return sign




def register(request): 
    if(request.method=="POST"):
        data= request.POST
        firstname = data.get('textfirstname')
        lastname=data.get('textlastname')
        username=data.get('textusername')
        password= data.get('textpassword')
        user=User.objects.filter(username=username)
        if(user.exists()):
            result="User name already exists"
            return render(request,'register.html',context={'result':result})

        if('submit' in request.POST ):
            user= User.objects.create(first_name=firstname,last_name=lastname,username=username)
            user.set_password(password)
            user.save()
            result="Registered successfully"
            return render(request,'register.html',context={'result':result})     
    return render(request,'register.html')



        

def registeration(request):
    if request.method=="POST":
        data=request.POST
        firstname=data.get('textfirstname')
        lastname=data.get('textlastname')
        username=data.get('textusername')
        password=data.get('textpassword')
        user=User.objects.filter(username=username)
        if(user.exists()):
            result="User name already exists"
            return render(request,'register.html',context={'result':result})
        user= User.objects.create(first_name=firstname,last_name=lastname,username=username,)
        user.set_password(password)
        user.save()
        result="Registration successfull"
        return render(request,'registeration.html',context={'result':result})
    return render(request,'registeration.html')

def home(request):
    return render(request,'home.html')





