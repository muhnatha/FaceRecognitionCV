import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.mean_face = np.mean(x, axis=0)
        return self
    
    def transform(self, x):
        return x-self.mean_face

# define model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# detect face funct
def detect_faces(img, scale_factor=1.1, min_neighbors = 5, min_size=(30, 30)):
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor = scale_factor,
        minNeighbors = min_neighbors,
        minSize = min_size,
    )
    return faces

# crop face funt
def crop_faces(img, faces, return_all = False):
    cropped_faces = []
    selected_faces = []
    if len(faces)>0:
        if return_all:
            for x, y, w, h in faces:
                selected_faces.append((x, y, w, h))
                cropped_faces.append(img[y:y+h, x:x+w])
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
            selected_faces.append((x, y, w, h))
            cropped_faces.append(img[y:y+h, x:x+w])
    return cropped_faces, selected_faces

# resize and Flatten
face_size = (128, 128)

def resize_and_flatten(face):
    face_resized = cv2.resize(face, face_size)
    face_flattened = face_resized.flatten()
    return face_flattened

# Combine
with open('eigenfaces_pipeline.pkl','rb') as f:
    pipe = pickle.load(f)

def get_eigenface_score(x):
    x_pca = pipe[:2].transform(x)
    eigenface_scores = np.max(pipe[2].decision_function(x_pca), axis=1)
    return eigenface_scores

def eigenface_prediction(gray):
    faces = detect_faces(gray)
    cropped_faces, selected_faces = crop_faces(gray, faces)

    if len(cropped_faces) == 0:
        return 'No face detected.'
    
    x_face = []
    for face in cropped_faces:
        face_flattened = resize_and_flatten(face)
        x_face.append(face_flattened)
    x_face = np.arrat(x_face)
    labels = pipe.predict(x_face)
    scores = get_eigenface_score(x_face)

    return scores, labels, selected_faces

def draw_text(img, label, score,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              pos=(0, 0),
              font_scale = 0.6,
              font_thickness = 2,
              text_color = (0, 0, 0),
              text_color_bg = (0, 255, 0)):
    
    x, y = pos
    score_text = f'Score: {score:.2f}'
    (w1, h1), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(img, (x, y-h1-h2-25), (x+max(w1, w1)+20, y), text_color_bg, -1)
    cv2.putText(img, label, (x+10, y-10), font, font_scale, text_color, font_thickness)
    cv2.putText(img, score_text, (x+10, y-h2-15), font, font_scale, text_color, font_thickness)

def draw_result(img, scores, labels, coords):
    result_img = img.copy()
    for(x, y, w, h), label, score, in zip(coords, labels, scores):
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        draw_text(result_img, label, score, pos = (x, y))
    return result_img

def runEigenface(img, gray):
    faces = detect_faces(gray)
    cropped_faces, selected_faces = crop_faces(gray, faces)

    if not cropped_faces:
        return 'No face detected.'
    
    x = np.array([resize_and_flatten(face) for face in cropped_faces])

    labels = pipe.predict(x)

    scores = np.max(pipe.decision_function(x), axis = 1)

    return scores, labels, selected_faces

def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result = runEigenface(frame, gray)
        if not isinstance(result, str):
            scores, labels, coords = result
            frame = draw_result(frame, scores, labels, coords)

        cv2.imshow('Live Eigenface', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    img_path = 'dataset/images/Cristiano_Ronaldo/01.jpg'
    process_webcam()


if __name__ == "__main__":  
    main()
