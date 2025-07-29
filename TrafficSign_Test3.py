import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pyttsx3 # sound audio
from threading import Thread # run proccess without blocking

#############################################
 
frameWidth =640         # CAMERA RESOLUTION
frameHeight =480
brightness = 180
threshold = 0.80      # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

##################TTS CONVERTER
# Initiate Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
 
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)


# Load your Keras trained model
model = load_model('traffic_model.keras')
#model = keras.models.load_model('traffic_classifier.h5')
#model = keras.models.load_model(f"traffic_detector.h5")
 
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def apply_canny(img):
    edges = cv2.Canny(img, 100, 200)  # Adjust thresholds as needed
    return edges

def preprocessing(img):
    img_gray = grayscale(img)
    img_equalized = equalize(img_gray)
    img_edges = apply_canny(img_equalized)
    img_edges = img_edges / 255.0  # Normalize
    return img_edges

classes = {
    
    0:['Speed limit (20km/h)', 'You should stay below 20 killometers per hour'],
    1:['Speed limit (30km/h)', 'You should stay below 30 killometers per hour'],
    2:['Speed limit (50km/h)', 'You should stay below 50 killometers per hour'],
    3:['Speed limit (60km/h)', 'You should stay below 60 killometers per hour'],
    4:['Speed limit (70km/h)', 'You should stay below 70 killometers per hour'],
    5:['Speed limit (80km/h)', 'You should stay below 80 killometers per hour'],
    6:['End of speed limit (80km/h)', 'You should stay below 80 killometers per hour'],
    7:['Speed limit (100km/h)', 'You should stay below 100 killometers per hour'],
    8:['Speed limit (120km/h)', 'You should stay below 120 killometers per hour'],
    9:['No passing', 'You should not pass'],
    10:['No passing veh over 3.5 tons', 'You should not pass if the vehicle is over 3.5 tons'],
    11:['Right-of-way at intersection', 'You should stay at the right of the road'],
    12:['', ''],
    13:['', ''],
    14:['Stop', 'You should stop the vehicle'],
    15:['No vehicles', 'No vehicles are allowed on this road'],
    16:['Veh > 3.5 tons prohibited', 'You should not pass if the vehicle is over 3.5 tons'],
    17:['No entry', 'You should not enter'],
    18:['General caution', 'General caution'],
    19:['Dangerous curve left', 'There is a dangerous curve to the left'],
    20:['Dangerous curve right', 'There is a dangerous curve to the right'],
    21:['Double curve', 'There is a double curve ahead'],
    22:['Bumpy road', 'There is a bumpy road ahead'],
    23:['Slippery road', 'There is a slippery road ahead'],
    24:['Road narrows on the right', 'The road is narrowing on the right'],
    25:['Road work', 'There is road work going on ahead'],
    26:['Traffic signals', 'There is a traffic signal ahead','Keep caution'],
    27:['Pedestrians', 'This is a pedestrian area'],
    28:['Children crossing', 'THIS IS A CHILDREN CROSSING AREA. \n Watch out for children \n Reduce Speed  \n Obey Any Signals From a Crossing Guard'],
    29:['Bicycles crossing', 'This is a bicycle area'],
    30:['Beware of ice/snow', 'There might be ice or snow ahead'],
    31:['Wild animals crossing', 'There might be wild animals ahead'],
    32:['End speed + passing limits', 'You should stay below 80 killometers per hour and not pass'],
    33:['Turn right ahead', 'You should turn right ahead'],
    34:['Turn left ahead', 'You should turn left ahead'],
    35:['Ahead only', 'You should not turn left or right ahead'],
    36:['Go straight or right', 'You should go straight or right and not turn left'],
    37:['Go straight or left', 'You should go straight or left and not turn right'],
    38:['Keep right', 'You should keep right and not turn left'],
    39:['Keep left', 'You should keep left and not turn right'],
    40:['Roundabout mandatory', 'You should go around the roundabout'],
    41:['End of no passing', 'You should not pass'],
    42:['End no passing vehicle with a weight greater than 3.5 tons', 'You should not pass if the vehicle is over 3.5 tons'],
    43:['No traffic sign', 'Traffic sign not detected, Please upload another image']
}

while True:
    success, imgOrignal = cap.read()

 
# PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)  
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "SIGN:        " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
# PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = np.argmax(model.predict(img))
    #classIndex = model.predict_step(img)
    probabilityValue = np.amax(predictions)
    className = classes[classIndex]
    if probabilityValue > threshold:
      print(className)
      print(probabilityValue)
      cv2.putText(imgOrignal,', ' +str(className[0]), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.putText(imgOrignal,str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
      tts_engine.say([className[0] + ', ' + className[1]])
##      Run without blocking by running it in a new thread
      Thread(target=tts_engine.runAndWait, daemon=True).start()
    cv2.imshow("Result", imgOrignal)  

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
