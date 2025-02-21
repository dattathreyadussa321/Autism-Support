import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from mtcnn import MTCNN
import os
import glob
import soundfile
import librosa
import librosa.display
import pyaudio
import wave
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

# Load the emotion detection model
emotion_model = load_model(r'C:/Users/Dattu/OneDrive/Desktop/Hackthon/emotion_model.keras')

# Emotion labels for face detection
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']

# Initialize MTCNN face detector
detector = MTCNN()

# Emotion Mapping for audio
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Feature Extraction for audio
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_features))

        if mel:
            mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_features))

        return result

# Real-Time Audio Recording
def record_audio(output_file="output.wav", record_seconds=4, rate=44100, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
    print("* Recording...")
    frames = [stream.read(chunk) for _ in range(0, int(rate / chunk * record_seconds))]
    print("* Done Recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Load the audio emotion model
def load_audio_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Predict emotion from audio
def predict_audio_emotion(model, audio_file):
    feature = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1, -1)
    prediction = model.predict(feature)
    return prediction[0]

# Preprocess the face for emotion recognition
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
    face = np.reshape(face, (1, 48, 48, 3))
    face = face.astype('float32') / 255
    return face

# Detect and predict emotion from the face
def detect_face_emotion(frame):
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        face_roi = frame[y:y+h, x:x+w]
        face_preprocessed = preprocess_face(face_roi)
        emotion_prediction = emotion_model.predict(face_preprocessed)
        max_index = np.argmax(emotion_prediction[0])
        emotion = emotion_labels[max_index]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Main Workflow
if __name__ == "__main__":
    # Load audio emotion model
    audio_model = load_audio_model(r'C:\Users\Dattu\Downloads\hack2\Emotion_Voice_Detection_Model.pkl')

    # Video Capture
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and predict emotion from video (face)
        frame = detect_face_emotion(frame)
        
        # Display the frame with detected emotion
        cv2.imshow('Emotion Detection', frame)
        
        # Predict emotion from audio
        record_audio(output_file="output.wav", record_seconds=4)
        audio_emotion = predict_audio_emotion(audio_model, "output.wav")
        print(f"Predicted Emotion from Audio: {audio_emotion}")
        
        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
