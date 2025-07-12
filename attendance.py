import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import time
from datetime import datetime
import sklearn 
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import RandomSampler





class FaceAttendanceSystem:
    
    def __init__(self):
        self.fm = mp.solutions.face_mesh
        self.model = self.fm.FaceMesh(static_image_mode=False,
                                     max_num_faces=1,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5,
                                     refine_landmarks=False)
        self.login_dt = None
        self.login_status = None
        self.logged_name = None
        
        
        #load the data 
        self.df = pd.read_csv(r"C:\Users\deepthi\Downloads\HostelFaces.csv")
        self.fv = self.df.iloc[:,:-1]
        self.cv = self.df.iloc[:,-1]
        self.final_pre_data = []
        
        for i in self.fv.values:
            self.md = i.reshape(468,3)
            self.center = self.md - i.reshape(468,3)[0]
            self.distance = np.linalg.norm(i.reshape(468,3)[33] - i.reshape(468,3)[263])
            self.fdp = self.center / self.distance
            self.final_pre_data.append(self.fdp.flatten())
    
        rf = RandomForestClassifier(n_estimators= 115,
                                     max_depth= 18,
                                     min_samples_split = 3,
                                     min_samples_leaf=3)
        self.rf_model = rf.fit(self.final_pre_data,self.cv)
        
    
    def register(self,name):
        if name in self.df['name'].values:
            print(f"{name} already exits in the dataset.")
            return
        vid = cv2.VideoCapture(1)
        df1 = pd.DataFrame(columns = [f"{i}" for i in range(1404)] + ["name"] )
     
        while True:
            s,frames = vid.read()
            if s == False:
                break
            rgb = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
            output = self.model.process(rgb)
            
            if output.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image = frames,
                                                          landmark_drawing_spec=None,
                                                         landmark_list= output.multi_face_landmarks[0],
                                                         connections = self.fm.FACEMESH_TESSELATION,
                                                         connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                cv2.imshow("Registration",frames)
                if cv2.waitKey(1) & 255 == ord("s"):
                    print(f"Starting capture for {name}.....")
               
                    samples_collected = 0
                    while samples_collected <= 200:
                        s,frames = vid.read()
                        if s == False:
                            break
                        rgb = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
                        output = self.model.process(rgb)
                        if output.multi_face_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(image=frames,landmark_list=output.multi_face_landmarks[0],
                                                                      landmark_drawing_spec=None,
                                                                      connections=self.fm.FACEMESH_TESSELATION,
                                                                     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
                            face = []
        
                            for idx in range(468):
                                lm = output.multi_face_landmarks[0].landmark[idx]
                                face.extend([lm.x, lm.y, lm.z] )
                            face.append(name)
        
                            df1.loc[len(df1)] = face
                            
                            samples_collected += 1
                            cv2.putText(frames, f"Recording {name}: {samples_collected}/200", (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        else:
                            cv2.putText(frames, "Face not detected", (30, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Registration",frames)
                        if cv2.waitKey(1) & 255 == ord("c"):
                            break
                    
                    self.df = pd.concat([self.df, df1], ignore_index=True)
                    print(f"Done collecting 200 samples for {name}")
                if cv2.waitKey(1) & 255 == ord("c"):
                    break
        vid.release()
        cv2.destroyAllWindows()
    def login(self):
        vid = cv2.VideoCapture(1)
        while True:
            s,frames = vid.read()
            if s == False:
                break
            rgb = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
            output = self.model.process(rgb)
            if output.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image = frames,
                                                          landmark_drawing_spec=None,
                                                         landmark_list = output.multi_face_landmarks[0],
                                                         connections = self.fm.FACEMESH_TESSELATION,
                                                         connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
            cv2.imshow("Login",frames)
            key = cv2.waitKey(1) & 255 
            if key == ord("i"):
                landmarks = output.multi_face_landmarks[0].landmark
                coords = np.array([[lm.x,lm.y,lm.z] for lm in landmarks])
                center = coords - coords[0]
                distance = np.linalg.norm(coords[33] - coords[263])
                fdp = center / distance
                features = fdp.flatten().reshape(1,-1)
                
                pred = self.rf_model.predict(features)[0]
                self.logged_name  = pred
                self.login_dt = datetime.now()
                self.login_status = "Login" if self.login_dt.hour <= 9 else "Late Login"
                self.login_time = self.login_dt.strftime("%H:%M:%S")
                print(f"{self.logged_name} had Logged in")
                break
            elif key == ord("c"):
                break
        vid.release()
        cv2.destroyAllWindows()

    def logout(self):
        
        vid = cv2.VideoCapture(1)
        while True:
            s,frames = vid.read()
            if s == False:
                break
            rgb = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
            output = self.model.process(rgb)
            
            if output.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image = frames,
                                                          landmark_drawing_spec=None,
                                                         landmark_list = output.multi_face_landmarks[0],
                                                         connections = self.fm.FACEMESH_TESSELATION,
                                                         connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
            cv2.imshow("Logout",frames)
            if cv2.waitKey(1) & 255 == ord("o"):
                landmarks = output.multi_face_landmarks[0].landmark
                coords = np.array([[lm.x,lm.y,lm.z] for lm in landmarks])
                center = coords - coords[0]
                distance = np.linalg.norm(coords[33] - coords[263])
                fdp = center / distance
                features = fdp.flatten().reshape(1,-1)
                
                pred = self.rf_model.predict(features)[0]
                logout_dt = datetime.now()
                logout_time = logout_dt.strftime("%H:%M:%S")
                working_hours = logout_dt - self.login_dt
                print(f"{self.logged_name} had logged out")
                
                attendance = pd.DataFrame([{"Name":self.logged_name,
                                           "LoginTime":self.login_time,
                                           "LoginStatus":self.login_status,
                                           "LogoutTime":logout_time,
                                           "Working_Hours":str(working_hours).split(".")[0]}])
                attendance.to_csv("Attendance System.csv", mode = 'a',header = not pd.io.common.file_exists("Attendance System.csv"), index=False)
                break
            elif cv2.waitKey(1) & 255 == ord("c"):
                break
        vid.release()
        cv2.destroyAllWindows()





system = FaceAttendanceSystem()

print("\n👋 --- FACE ATTENDANCE MENU --- 👇\n"
      "1️⃣ Register 📝\n"
      "2️⃣ Login 🔐\n"
      "3️⃣ Logout 🚪\n"
      "4️⃣ Exit ❌")

while True:
    choice = input("👉 Choose an option (1-4): ")

    if choice == '1':
        name = input("🧑 Enter your name: ")
        system.register(name)
    elif choice == '2':
        system.login()
    elif choice == '3':
        system.logout()
    elif choice == '4':
        print("👋 Exiting... Have a great day!")
        break
    else:
        print("⚠️ Invalid option. Please try again.")
