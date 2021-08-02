from keras.models import load_model
import cv2
import numpy as np
import tkinter
from tkinter import messagebox
import smtplib
import os
from numpy.lib.utils import source



if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')


root = tkinter.Tk()
root.withdraw()

model = load_model('C:\\Users\\NELSON JOSEPH\\Desktop\\programs\\face_mask_detection_alert_system.h5')

face_det_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#vid_source = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid_source = cv2.VideoCapture(0)


# text_dict = {0:'Mask ON', 1: 'No Mask'}
# rect_color_dict = {0:(0,255,0), 1:(0,0,255)}

text_dict={0:'Mask ON',1:'No Mask'}
rect_color_dict={0:(0,255,0),1:(0,0,255)}  #BGR

SUBJECT = "Subject"
TEXT = "One visitor violated Face Mask Policy. See in Camera to recognize the user"

# while (True):
#     ret, img = vid_source.read()
#     grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_det_classifier.detectMultiScale(grayscale_img,1.3,5)

#     for (x,y,w,h) in faces:
#         face_img = grayscale_img[y:y+w, x:x+w]
#         resized_img = cv2.resize(face_img, (56,56))
#         normalized_img = resized_img/ 255.0
#         reshaped_img = np.reshape(normalized_img, (1,56,56,1)) #(1,112,112,1)
#         result = model.predict(reshaped_img)

#         label = np.argmax(result, axis = 1)[0]
#         cv2.rectangle(img,(x,y),(x+w,y+h),rect_color_dict[label],2)
#         cv2.rectangle(img,(x,y-40),(x+w,y),rect_color_dict[label],-1)
#         cv2.putText(img, text_dict[label], (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

#         if (label == 1):

#             messagebox.showwarning("warning","Access Denied. Please wear a Mask")

#             message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)
#             mail = smtplib.SMTP('smtp.gmail.com', 587)
#             mail.eclo()
#             mail.starttls()
#             mail.login('nelsonjoseph286@gmail.com','Qwerty<>!')
#             mail.senmail('nelsonjoseph123@gmail.com','nelsonjoseph123@gmail.com',message)
#             mail.close
#         else:
#             pass
#             break
#     cv2.imshow('LIVE Video Feed', img)
#     key = cv2.waitKey(1)

#     if (key ==27):
#         break

# cv2.destroyAllWindows()
# cv2.realse()

while(True):

    ret, img = vid_source.read()
    grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_det_classifier.detectMultiScale(grayscale_img,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img = grayscale_img[y:y+w,x:x+w]
        resized_img = cv2.resize(face_img,(112,112))
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img,(1,112,112,1))
        result=model.predict(reshaped_img)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),rect_color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),rect_color_dict[label],-1)
        cv2.putText(img, text_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2) 
        
        # If label = 1 then it means wearing No Mask and 0 means wearing Mask
        if (label == 1):
            # Throw a Warning Message to tell user to wear a mask if not wearing one. This will stay
            #open and No Access will be given He/She wears the mas
            messagebox.showwarning("warning","Access Denied. Please wear a Mask",)
            # label1 = tkinter.Label(root, text="\n\n\n\nAccess Denied. Please wear a Face Mask", font=("Helvetica", 16))
            
            # Send an email to the administrator if access denied/user not wearing face mask 
            message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)
            mail = smtplib.SMTP('smtp.gmail.com', 587)
            mail.ehlo()
            mail.starttls()
            mail.login('xyz@gmail.com','password')
            mail.sendmail('xyz@gmail.com','xyz@gmail.com',message)
            mail.close
        else:
            print("mask")
            

    cv2.imshow('LIVE Video Feed',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()





