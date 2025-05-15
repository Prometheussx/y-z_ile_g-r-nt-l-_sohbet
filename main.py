from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import threading
import time
import base64
import numpy as np
import os
from dotenv import load_dotenv
import requests
import json
import pyaudio
import wave
import webrtcvad
import warnings
# Suppress audioop deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop
import tempfile
from openai import OpenAI
from io import BytesIO
import queue
import socketio
import concurrent.futures
from functools import lru_cache
import urllib.request
import ssl
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Flask ve SocketIO yapılandırması
flask_app = Flask(__name__)
sio = socketio.Server(async_mode='threading', cors_allowed_origins='*')
app = socketio.WSGIApp(sio, flask_app)

# Global variables
lock = threading.Lock()
frame_buffer = None
last_processed_time = 0
processing_interval = 2  # Reduced from 3 seconds to 2 seconds
is_listening = False     # Active listening off by default
is_speaking = False      # Is the assistant currently speaking
is_passive_listening = False  # Passive listening initially off
audio_queue = queue.Queue()
RATE = 16000
CHUNK = 320  # 20ms at 16kHz (WebRTC VAD requires 10, 20, or 30 ms frames)
FORMAT = pyaudio.paInt16
CHANNELS = 1
VAD_MODE = 3  # VAD aggressiveness (0-3)

# Sound detection thresholds
ENERGY_THRESHOLD = 450  # Minimum energy level to consider as sound
PASSIVE_ENERGY_THRESHOLD = 350  # Lower threshold for passive detection
SOUND_DURATION_THRESHOLD = 4  # Required consecutive chunks with sound to activate (~80ms)

# AI Assistant Configuration
AI_NAME = "Temel"  # The name of our AI assistant

# Threading Pools for parallel processing
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
api_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# API Call Cache
IMAGE_CACHE_SIZE = 5
API_CACHE_SIZE = 20

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Voice Activity Detection
vad = webrtcvad.Vad(VAD_MODE)

# Audio processing thread variables
stop_listening_event = threading.Event()
pause_listening_event = threading.Event()
passive_stop_event = threading.Event()
audio_processing_thread = None
passive_listening_thread = None

# Keep track of the latest TTS audio file
latest_tts_file = None

# Client frame storage - key is the client session ID, value is their latest frame
client_frames = {}
client_processed_frames = {}  # Store frames with face recognition visualization
# Default client sid for thread context where request is not available
default_client_sid = None

# -------------------------------------------------------------------------------
# Face Recognition Implementation
# -------------------------------------------------------------------------------

class FaceRecognizer:
    def __init__(self, reference_images_dir='static/faces'):
        """
        Initialize face recognition with the directory containing reference face images.
        Each person should have their own subfolder with their name and one image inside.
        
        Args:
            reference_images_dir: Directory containing person folders with reference images
        """
        # Create faces directory if it doesn't exist
        if not os.path.exists(reference_images_dir):
            os.makedirs(reference_images_dir)
            
        self.reference_dir = reference_images_dir
        
        # Load face detection model
        haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(haar_cascade_path)
        
        # Load face recognition model (deep learning based)
        # First, download the model if it doesn't exist
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "openface_nn4.small2.v1.t7")
        
        if not os.path.exists(model_path):
            print("Face recognition model not found, downloading...")
            self._download_model(model_path)
            
        # Initialize face recognizer
        try:
            self.face_recognizer = cv2.dnn.readNetFromTorch(model_path)
            print("Face recognition model loaded successfully")
        except Exception as e:
            print(f"Error loading face recognition model: {e}")
            # Try to download again if loading fails
            if os.path.exists(model_path):
                os.remove(model_path)
            self._download_model(model_path)
            try:
                self.face_recognizer = cv2.dnn.readNetFromTorch(model_path)
                print("Face recognition model loaded successfully on second attempt")
            except Exception as e:
                print(f"Failed to load face recognition model: {e}")
                # Create an empty network as fallback
                self.face_recognizer = cv2.dnn.Net()
        
        # Dictionary to store reference face embeddings
        self.known_face_encodings = {}
        self.known_face_names = []
        
        # Load any pre-existing face embeddings
        self.embedding_file = os.path.join(reference_images_dir, "embeddings.pkl")
        self.load_face_embeddings()
    
    def _download_model(self, model_path):
        """Download the OpenFace model"""
        try:
            print("Downloading face recognition model...")
            ssl_context = ssl._create_unverified_context()
            model_url = "https://github.com/pyannote/pyannote-data/raw/master/openface.nn4.small2.v1.t7"
            
            with urllib.request.urlopen(model_url, context=ssl_context) as response:
                with open(model_path, 'wb') as f:
                    f.write(response.read())
            print("Face recognition model downloaded successfully")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please download the model manually and place it in the 'models' directory")
        
    def load_face_embeddings(self):
        """Load saved face embeddings if they exist"""
        if os.path.exists(self.embedding_file):
            try:
                with open(self.embedding_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', {})
                    self.known_face_names = data.get('names', [])
                print(f"Loaded {len(self.known_face_names)} face profiles")
            except Exception as e:
                print(f"Error loading face embeddings: {e}")
                # If there's an error loading embeddings, try rebuilding the database
                self.build_face_database()
        else:
            # Create embeddings for any reference images
            self.build_face_database()
            
    def save_face_embeddings(self):
        """Save face embeddings to file"""
        try:
            os.makedirs(os.path.dirname(self.embedding_file), exist_ok=True)
            with open(self.embedding_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            print(f"Saved {len(self.known_face_names)} face profiles")
        except Exception as e:
            print(f"Error saving face embeddings: {e}")

    def build_face_database(self):
        """Build face database from reference images in the faces directory"""
        print("Building face database from reference images...")
        try:
            # Clear existing data
            self.known_face_encodings = {}
            self.known_face_names = []
            
            # Look for direct image files in the faces directory
            for file_name in os.listdir(self.reference_dir):
                file_path = os.path.join(self.reference_dir, file_name)
                
                # Skip directories and non-image files
                if os.path.isdir(file_path) or not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                # Get person name from file name (remove extension)
                person_name = os.path.splitext(file_name)[0]
                print(f"Processing reference image for {person_name}")
                
                # Load the image
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Could not read image {file_path}")
                    continue
                
                # Detect face in the reference image
                face = self.detect_face(img)
                if face is None:
                    print(f"No face detected in {file_path}")
                    continue
                
                # Extract face embedding
                face_encoding = self.get_face_embedding(face)
                
                # Store in our database
                self.known_face_encodings[person_name] = face_encoding
                self.known_face_names.append(person_name)
            
            # Also check subdirectories (for backward compatibility with previous version)
            for person_name in os.listdir(self.reference_dir):
                person_dir = os.path.join(self.reference_dir, person_name)
                if not os.path.isdir(person_dir):
                    continue
                    
                print(f"Processing reference images for {person_name}")
                
                # Get all image files in person's directory
                image_files = [f for f in os.listdir(person_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not image_files:
                    print(f"No images found for {person_name}")
                    continue
                
                # Use the first image as reference
                image_path = os.path.join(person_dir, image_files[0])
                img = cv2.imread(image_path)
                
                if img is None:
                    print(f"Could not read image {image_path}")
                    continue
                
                # Detect face in the reference image
                face = self.detect_face(img)
                if face is None:
                    print(f"No face detected in {image_path}")
                    continue
                
                # Extract face embedding
                face_encoding = self.get_face_embedding(face)
                
                # Store in our database
                self.known_face_encodings[person_name] = face_encoding
                self.known_face_names.append(person_name)
                
            self.save_face_embeddings()
            print(f"Face database built with {len(self.known_face_names)} people")
            
        except Exception as e:
            print(f"Error building face database: {e}")

    def detect_face(self, frame):
        """
        Detect a face in the frame and return the face image
        Returns None if no face found
        """
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, 
                                                      minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                return None
            
            # Use the largest face detected (closest to camera)
            face_areas = [w * h for (x, y, w, h) in faces]
            largest_face_idx = np.argmax(face_areas)
            x, y, w, h = faces[largest_face_idx]
            
            # Extract face with some margin
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w += 2 * margin
            h += 2 * margin
            
            # Make sure we don't go out of frame bounds
            height, width = frame.shape[:2]
            x_end = min(width, x + w)
            y_end = min(height, y + h)
            
            # Extract the face region
            face_img = frame[y:y_end, x:x_end]
            
            return face_img
        except Exception as e:
            print(f"Error in detect_face: {e}")
            return None
        
    def detect_all_faces(self, frame):
        """
        Detect all faces in the frame and return their coordinates
        Returns list of (x, y, w, h) tuples
        """
        try:
            # Make sure the frame is valid
            if frame is None or frame.size == 0:
                print("Invalid frame in detect_all_faces")
                return []
                
            # Convert to grayscale for face detection
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Error converting frame to grayscale: {e}")
                return []
            
            # Detect faces
            try:
                faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, 
                                                          minNeighbors=5, minSize=(30, 30))
            except Exception as e:
                print(f"Error in face detection: {e}")
                return []
            
            return faces
        except Exception as e:
            print(f"Error in detect_all_faces: {e}")
            return []
        
    def get_face_embedding(self, face_img):
        """Extract the 128-d face embedding using deep learning model"""
        try:
            if face_img is None or face_img.size == 0:
                return np.zeros(128)  # Return empty embedding
                
            # Resize to required input size
            face_blob = cv2.dnn.blobFromImage(face_img, 1.0/255, (96, 96), 
                                             (0, 0, 0), swapRB=True, crop=False)
            
            # Pass through the network
            self.face_recognizer.setInput(face_blob)
            embedding = self.face_recognizer.forward()
            
            # Normalize embedding vector
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error in get_face_embedding: {e}")
            return np.zeros(128)  # Return empty embedding
        
    def recognize_face(self, frame, threshold=0.6):
        """
        Recognize faces in the frame
        
        Args:
            frame: Video frame
            threshold: Similarity threshold for recognition (0-1)
            
        Returns:
            (name, confidence) or (None, 0) if no face or match found
        """
        try:
            # Detect face
            face = self.detect_face(frame)
            
            if face is None:
                return None, 0
            
            # Calculate face embedding
            face_encoding = self.get_face_embedding(face)
            
            # Check if we have any known faces
            if not self.known_face_encodings:
                return None, 0
                
            # Find the best match
            best_match = None
            best_score = 0
            
            for name, ref_encoding in self.known_face_encodings.items():
                # Calculate cosine similarity
                similarity = cosine_similarity([face_encoding], [ref_encoding])[0][0]
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = name
            
            # Return match if confidence is above threshold
            if best_score >= threshold:
                return best_match, best_score
            else:
                return None, best_score
        except Exception as e:
            print(f"Error in recognize_face: {e}")
            return None, 0
    
    def recognize_faces_in_frame(self, frame, threshold=0.6):
        """
        Recognize all faces in the frame
        
        Args:
            frame: Video frame
            threshold: Similarity threshold for recognition (0-1)
            
        Returns:
            List of (name, confidence, x, y, w, h) tuples
        """
        results = []
        try:
            # Make sure frame is valid
            if frame is None or frame.size == 0:
                print("Invalid frame in recognize_faces_in_frame")
                return results
                
            # Detect all faces
            faces = self.detect_all_faces(frame)
            
            # Process each face
            for (x, y, w, h) in faces:
                try:
                    # Extract face region with some margin
                    margin = 5
                    x_start = max(0, x - margin)
                    y_start = max(0, y - margin)
                    x_end = min(frame.shape[1], x + w + margin)
                    y_end = min(frame.shape[0], y + h + margin)
                    
                    face_img = frame[y_start:y_end, x_start:x_end]
                    
                    if face_img.size == 0:
                        continue
                    
                    # Get face embedding
                    face_encoding = self.get_face_embedding(face_img)
                    
                    # Find the best match
                    best_match = None
                    best_score = 0
                    
                    for name, ref_encoding in self.known_face_encodings.items():
                        # Calculate cosine similarity
                        similarity = cosine_similarity([face_encoding], [ref_encoding])[0][0]
                        
                        if similarity > best_score:
                            best_score = similarity
                            best_match = name
                    
                    # Add result if confidence is above threshold
                    if best_score >= threshold:
                        results.append((best_match, best_score, x, y, w, h))
                    else:
                        results.append(("Unknown", best_score, x, y, w, h))
                except Exception as e:
                    print(f"Error processing a face in the frame: {e}")
                    continue
            
            return results
        except Exception as e:
            print(f"Error in recognize_faces_in_frame: {e}")
            return results

    def draw_face_recognition(self, frame):
        """Draw face recognition results on frame"""
        try:
            # Make a copy of the frame to avoid modifying the original
            if frame is None or frame.size == 0:
                print("Invalid frame in draw_face_recognition")
                return None
                
            display_frame = frame.copy()
            
            # Recognize all faces in the frame
            results = self.recognize_faces_in_frame(display_frame)
            
            # Draw rectangles and labels for each face
            for (name, confidence, x, y, w, h) in results:
                # Draw rectangle around face
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for recognized, red for unknown
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                # Create label with name and confidence
                label = f"{name} ({confidence:.2f})"
                
                # Draw label background
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_frame, 
                             (x, y - label_size[1] - 10),
                             (x + label_size[0], y),
                             color, 
                             cv2.FILLED)
                
                # Draw label text
                cv2.putText(display_frame, label, (x, y - 7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return display_frame
        except Exception as e:
            print(f"Error in draw_face_recognition: {e}")
            return frame  # Return original frame if error

# Initialize face recognizer
face_recognizer = FaceRecognizer()

# Helper functions for face recognition
def recognize_face_in_frame(frame):
    """Recognize a face in the frame and return name and confidence"""
    name, confidence = face_recognizer.recognize_face(frame)
    return name, confidence

def recognize_all_faces_in_frame(frame):
    """Recognize all faces in the frame"""
    return face_recognizer.recognize_faces_in_frame(frame)

def get_faces_with_recognition(frame):
    """Get frame with face recognition visualization"""
    return face_recognizer.draw_face_recognition(frame)

# -------------------------------------------------------------------------------
# End of Face Recognition Implementation
# -------------------------------------------------------------------------------

# Cache for repeated API calls with same inputs
@lru_cache(maxsize=API_CACHE_SIZE)
def _cached_analyze_frame(image_hash, prompt):
    """Cached version of frame analysis (called by analyze_frame_with_gpt)"""
    # This is just a helper function that can be cached
    pass

def analyze_frame_with_gpt(frame, query_text="What can you see in this image? If there's text, please read it."):
    """Send the frame to OpenAI's API for analysis"""
    try:
        # First, recognize faces in the frame
        results = recognize_all_faces_in_frame(frame)
        
        # Extract names of recognized people with good confidence
        recognized_people = []
        for name, confidence, _, _, _, _ in results:
            if name != "Unknown" and confidence > 0.6:
                recognized_people.append((name, confidence))
        
        # Personalized prompt construction based on recognized people
        prompt = ""
        
        # If we have recognized people, include their names in the prompt
        if recognized_people:
            main_person = recognized_people[0]  # Take the first recognized person as main addressee
            
            # Construct a personalized prompt for GPT-4o
            prompt = (f"You are {AI_NAME}, a helpful voice assistant. "
                      f"I can see {main_person[0]} in the camera view. Please address {main_person[0]} directly in your response. "
                      f"Be conversational and friendly. Respond to this query from {main_person[0]}: \"{query_text}\"\n\n"
                      f"Start your response with a greeting that includes {main_person[0]}'s name. "
                      f"Describe what you see them doing if relevant to their question. "
                      f"If they're making gestures or showing something, mention it.")
            
            # If there are additional people in the frame
            if len(recognized_people) > 1:
                others = [name for name, _ in recognized_people[1:]]
                others_str = ", ".join(others)
                prompt += f"\n\nI also see {others_str} in the frame. You may acknowledge them too if relevant."
        else:
            # General prompt if no one is recognized
            prompt = (f"You are {AI_NAME}, a helpful voice assistant. "
                      f"Answer this query: \"{query_text}\"\n\n"
                      f"Be conversational and friendly. Describe what you see in the image if relevant.")
        
        # Draw the face recognition results on the frame
        # This way GPT-4 can also see who is recognized in the image
        if results:
            # Make a copy of the frame to avoid modifying the original
            frame_with_faces = frame.copy()
            
            # Draw rectangles and labels for each face
            for (name, confidence, x, y, w, h) in results:
                # Draw rectangle around face
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for recognized, red for unknown
                cv2.rectangle(frame_with_faces, (x, y), (x+w, y+h), color, 2)
                
                # Create label with name and confidence
                label = f"{name} ({confidence:.2f})"
                
                # Draw label background
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_with_faces, 
                             (x, y - label_size[1] - 10),
                             (x + label_size[0], y),
                             color, 
                             cv2.FILLED)
                
                # Draw label text
                cv2.putText(frame_with_faces, label, (x, y - 7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            # Use this modified frame with face recognition visualization
            frame = frame_with_faces
        
        # Reduce image size for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Convert the OpenCV frame to a format suitable for the API
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return "Error encoding image"
        
        # Convert to base64 string
        image_binary = BytesIO(buffer).getvalue()
        
        # Create a hash of image + prompt for caching
        image_hash = hash(image_binary[:1000] + prompt.encode())  # Use partial image for hash
        
        # Check if we have this analysis cached
        if hasattr(_cached_analyze_frame, 'cache_info'):
            cache_info = _cached_analyze_frame.cache_info()
            if cache_info.hits > 0:
                cached_result = _cached_analyze_frame(image_hash, prompt)
                if cached_result:
                    print("Using cached analysis result")
                    return cached_result
        
        # Send to OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using gpt-4o which supports vision capabilities
            messages=[
                {
                    "role": "system",
                    "content": (f"You are {AI_NAME}, a friendly and helpful AI assistant who has access to camera. "
                                f"You can see through the camera and address people by name when you recognize them. "
                                f"Keep responses conversational and personal. When you see someone you recognize, "
                                f"address them directly by their name. If they are making gestures, point it out.")
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_binary).decode('utf-8')}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        
        result = response.choices[0].message.content
        
        # Update the cache
        _cached_analyze_frame.cache_clear()  # Clear old entries
        _cached_analyze_frame.__wrapped__ = lambda ih, p: result if ih == image_hash else None
        
        return result
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return f"Error analyzing frame: {str(e)}"

@lru_cache(maxsize=API_CACHE_SIZE)
def text_to_speech_openai(text, output_file=None):
    """Convert text to speech using OpenAI's TTS API"""
    global latest_tts_file
    
    try:
        # If no output file specified, create a temporary file
        if not output_file:
            temp_dir = "static/audio"
            # Create directory if it doesn't exist
            os.makedirs(temp_dir, exist_ok=True)
            output_file = f"{temp_dir}/tts_{int(time.time())}.mp3"
        
        # Check if we already have this text-to-speech conversion cached
        if os.path.exists(output_file):
            latest_tts_file = output_file
            return output_file
        
        # Call OpenAI TTS API
        response = client.audio.speech.create(
            model="tts-1", 
            voice="echo",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )

        # Save the audio file
        response.stream_to_file(output_file)
        
        # Update the latest TTS file for frontend access
        latest_tts_file = output_file
        
        return output_file
    except Exception as e:
        print(f"Error in text to speech: {e}")
        return None

def passive_listening_worker():
    """Background thread to monitor audio for sound activity"""
    global is_listening, is_passive_listening
    
    print("Starting passive listening mode...")
    is_passive_listening = True
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    # Variables for sound detection
    sound_detected_count = 0
    
    try:
        while not passive_stop_event.is_set():
            # Skip if we're actively processing speech or already in active mode
            if is_listening or is_speaking:
                time.sleep(0.1)
                sound_detected_count = 0
                continue
                
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            
            # Energy-based detection for passive mode
            rms = audioop.rms(audio_chunk, 2)
            
            if rms > PASSIVE_ENERGY_THRESHOLD:
                sound_detected_count += 1
                
                # If sound persists for enough chunks, activate full listening
                if sound_detected_count >= SOUND_DURATION_THRESHOLD:
                    print("Sound detected, activating listening mode...")
                    # Activate listening mode
                    start_listening()
                    sound_detected_count = 0
            else:
                sound_detected_count = max(0, sound_detected_count - 1)  # Decay counter if no sound
                
            time.sleep(0.01)  # Small sleep to reduce CPU usage
                
    except Exception as e:
        print(f"Error in passive listening: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        is_passive_listening = False
        print("Stopped passive listening")

def audio_processing_worker():
    """Background thread to continuously process audio"""
    global is_listening, is_speaking
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    print("Starting active voice detection...")
    
    # Variables for voice detection
    local_is_speaking = False
    silent_chunks = 0
    voice_chunks = 0
    max_silent_chunks = 40  # ~800ms of silence
    min_voice_chunks = 4    # ~80ms of voice
    accumulated_audio = []
    
    try:
        while not stop_listening_event.is_set():
            # Skip processing if we're paused (assistant is speaking)
            if pause_listening_event.is_set():
                time.sleep(0.1)  # Short sleep to avoid CPU spike
                continue
                
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            
            # Check if this chunk contains speech
            try:
                # Convert to proper format for WebRTC VAD
                is_speech = vad.is_speech(audio_chunk, RATE)
            except Exception as e:
                # If WebRTC VAD fails, fallback to energy-based detection
                rms = audioop.rms(audio_chunk, 2)
                is_speech = rms > ENERGY_THRESHOLD
            
            # State machine for speech detection
            if is_speech and not local_is_speaking:
                # Potential start of speech
                voice_chunks += 1
                accumulated_audio.append(audio_chunk)  # Start saving audio
                
                if voice_chunks >= min_voice_chunks:
                    print("Speech started")
                    local_is_speaking = True
            
            elif is_speech and local_is_speaking:
                # Continuing speech
                voice_chunks += 1
                silent_chunks = 0
                accumulated_audio.append(audio_chunk)
            
            elif not is_speech and local_is_speaking:
                # Potential end of speech
                silent_chunks += 1
                accumulated_audio.append(audio_chunk)
                
                if silent_chunks >= max_silent_chunks:
                    print(f"Speech ended, processing... (collected {len(accumulated_audio)} chunks)")
                    
                    # Process the accumulated audio if we have enough
                    if len(accumulated_audio) > 8:
                        # Convert list of chunks to single bytes object
                        audio_data = b''.join(accumulated_audio)
                        
                        # Pause listening while processing - will be resumed after TTS finishes
                        pause_listening_event.set()
                        is_speaking = True
                        # Notify frontend that we're processing speech
                        sio.emit('listening_status', {'is_listening': False, 'reason': 'processing'})
                        
                        # Process in a separate thread from the thread pool
                        thread_pool.submit(process_speech, audio_data)
                    else:
                        # If not enough speech, return to passive mode
                        stop_active_return_to_passive()
                    
                    # Reset state
                    local_is_speaking = False
                    silent_chunks = 0
                    voice_chunks = 0
                    accumulated_audio = []
            
            elif not is_speech and not local_is_speaking:
                # No speech detected
                voice_chunks = 0
                silent_chunks += 1
                
                # Keep a small buffer for better detection
                if len(accumulated_audio) > 8:
                    accumulated_audio = accumulated_audio[-8:]
                    
                # If extended silence after activation, return to passive mode
                if silent_chunks > max_silent_chunks * 2:
                    print("Extended silence, returning to passive mode")
                    stop_active_return_to_passive()
                    break
                
    except Exception as e:
        print(f"Error in audio processing: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Stopped active voice detection")
        # Make sure we return to passive listening if this thread ends
        ensure_passive_listening()

def process_speech(audio_data):
    """Process detected speech and get response"""
    global is_speaking
    
    try:
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_audio_path = temp_file.name
            
        # Write WAV file with proper headers
        with wave.open(temp_audio_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit audio = 2 bytes
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        
        # Send to Whisper API using thread pool
        print("Sending to Whisper API...")
        future = api_thread_pool.submit(transcribe_audio, temp_audio_path)
        query_text = future.result()
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        if not query_text or query_text.strip() == "":
            print("Empty transcription")
            # Return to passive mode
            stop_active_return_to_passive()
            return
            
        print(f"Transcribed: {query_text}")
        
        # Get current frame for analysis
        with lock:
            # Use default client sid if available
            if default_client_sid and default_client_sid in client_frames and client_frames[default_client_sid] is not None:
                frame_to_analyze = client_frames[default_client_sid].copy()
            # Otherwise use the first available client frame
            elif client_frames:
                # Use the most recent client's frame
                sid = list(client_frames.keys())[0]
                if client_frames[sid] is not None:
                    frame_to_analyze = client_frames[sid].copy()
                elif frame_buffer is not None:
                    frame_to_analyze = frame_buffer.copy()
                else:
                    print("No frame available")
                    stop_active_return_to_passive()
                    return
            # Fall back to the global frame buffer
            elif frame_buffer is not None:
                frame_to_analyze = frame_buffer.copy()
            else:
                print("No frame available")
                stop_active_return_to_passive()
                return
            
        # Analyze the frame with the query using thread pool
        print("Sending to GPT-4o for analysis...")
        future = api_thread_pool.submit(analyze_frame_with_gpt, frame_to_analyze, query_text)
        answer = future.result()
        print(f"GPT Response: {answer}")
        
        # Generate speech with OpenAI's TTS using thread pool
        future = api_thread_pool.submit(text_to_speech_openai, answer)
        tts_file = future.result()
        
        # Send result to frontend
        audio_queue.put({
            "type": "conversation",
            "query": query_text,
            "answer": answer,
            "audio_file": tts_file
        })
        
        # We'll keep listening paused until the frontend tells us audio playback is complete
        
    except Exception as e:
        print(f"Error in speech processing: {e}")
        audio_queue.put({
            "type": "error",
            "message": str(e)
        })
        # Return to passive mode in case of error
        stop_active_return_to_passive()

def transcribe_audio(audio_path):
    """Transcribe audio file using Whisper API"""
    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file
                # Letting Whisper auto-detect the language
            )
        return transcription.text
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""

def resume_listening():
    """Resume listening after TTS is done playing"""
    global is_speaking
    is_speaking = False
    pause_listening_event.clear()
    
    # KRITIK: Ses bittiğinde pasif moda geç
    print("Audio finished playing, returning to passive listening mode")
    stop_active_return_to_passive()

def stop_active_return_to_passive():
    """Stop active listening and return to passive mode"""
    global is_listening, is_speaking
    
    # Reset flags
    is_listening = False
    is_speaking = False
    
    # Stop active listening
    stop_listening()
    
    # Make sure passive listening is running
    ensure_passive_listening()
    
    # Notify frontend of mode change
    sio.emit('listening_status', {'is_listening': False, 'reason': 'passive'})

def ensure_passive_listening():
    """Ensure passive listening is running"""
    global is_passive_listening, passive_listening_thread
    
    if not is_passive_listening or passive_listening_thread is None or not passive_listening_thread.is_alive():
        start_passive_listening()

def start_listening():
    """Start the continuous listening thread"""
    global audio_processing_thread, stop_listening_event, is_listening
    
    if audio_processing_thread is not None and audio_processing_thread.is_alive():
        print("Already listening")
        return False
    
    # Make sure we stop passive listening first
    stop_passive_listening()
    
    stop_listening_event.clear()
    pause_listening_event.clear()
    audio_processing_thread = threading.Thread(target=audio_processing_worker)
    audio_processing_thread.daemon = True
    audio_processing_thread.start()
    is_listening = True
    sio.emit('listening_status', {'is_listening': True, 'reason': 'activated'})
    return True

def stop_listening():
    """Stop the continuous listening thread"""
    global audio_processing_thread, stop_listening_event, is_listening
    
    if audio_processing_thread is None or not audio_processing_thread.is_alive():
        print("Not currently listening")
        return False
    
    stop_listening_event.set()
    is_listening = False
    
    # Don't wait for the thread to terminate - it may block if reading from the stream
    # Instead let it finish naturally on its next iteration
    return True

def start_passive_listening():
    """Start the passive sound detection thread"""
    global passive_listening_thread, passive_stop_event, is_passive_listening
    
    if passive_listening_thread is not None and passive_listening_thread.is_alive():
        print("Passive listening already active")
        return False
    
    passive_stop_event.clear()
    passive_listening_thread = threading.Thread(target=passive_listening_worker)
    passive_listening_thread.daemon = True
    passive_listening_thread.start()
    is_passive_listening = True
    return True

def stop_passive_listening():
    """Stop the passive sound detection thread"""
    global passive_listening_thread, passive_stop_event, is_passive_listening
    
    if passive_listening_thread is None or not passive_listening_thread.is_alive():
        print("Passive listening not active")
        return False
    
    passive_stop_event.set()
    is_passive_listening = False
    
    # Don't wait for the thread to terminate - same reason as active listening
    return True

def pause_listening():
    """Temporarily pause listening"""
    global is_listening
    pause_listening_event.set()
    is_listening = False

# Process frames for face recognition
def process_frames_with_face_recognition():
    """Background thread to process frames with face recognition"""
    try:
        while True:
            # Process frames for each client
            with lock:
                for sid, frame in client_frames.items():
                    if frame is not None:
                        try:
                            # Process frame with face recognition
                            processed_frame = get_faces_with_recognition(frame)
                            if processed_frame is not None:
                                client_processed_frames[sid] = processed_frame
                        except Exception as e:
                            print(f"Error processing frame for client {sid}: {e}")
            
            # Sleep to avoid high CPU usage
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in process_frames_with_face_recognition: {e}")

# Start frame processing thread
face_recognition_thread = threading.Thread(target=process_frames_with_face_recognition)
face_recognition_thread.daemon = True
face_recognition_thread.start()

# Socket.IO event handlers
@sio.event
def connect(sid, environ):
    print(f"Client connected: {sid}")
    global default_client_sid
    # Set as default client if we don't have one yet
    if default_client_sid is None:
        default_client_sid = sid
    # Initialize client's frame storage
    client_frames[sid] = None
    client_processed_frames[sid] = None
    # Tell client to start sending camera frames
    sio.emit('request_camera', {}, room=sid)
    # Update client about current status
    if is_speaking:
        sio.emit('listening_status', {'is_listening': False, 'reason': 'speaking'}, room=sid)
    elif is_listening:
        sio.emit('listening_status', {'is_listening': True, 'reason': 'active'}, room=sid)
    elif is_passive_listening:
        sio.emit('listening_status', {'is_listening': False, 'reason': 'passive'}, room=sid)
    else:
        sio.emit('listening_status', {'is_listening': False, 'reason': 'inactive'}, room=sid)

@sio.event
def disconnect(sid):
    print(f"Client disconnected: {sid}")
    global default_client_sid
    # Clean up client's frame storage
    if sid in client_frames:
        del client_frames[sid]
    if sid in client_processed_frames:
        del client_processed_frames[sid]
    # Reset default client if this was the default
    if default_client_sid == sid:
        if client_frames:  # If there are other clients
            default_client_sid = next(iter(client_frames.keys()))
        else:
            default_client_sid = None
    
@sio.event
def audio_completed(sid, data):
    """Client notifies server that TTS audio playback is complete"""
    print("Audio playback complete, resuming listening")
    thread_pool.submit(resume_listening)

@sio.event
def client_frame(sid, data):
    """Receive camera frame from client"""
    try:
        # Decode base64 image from client
        image_data = data.get('image')
        if not image_data:
            return
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            
        # Decode base64 string
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image with OpenCV
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Failed to decode frame")
            return
            
        # Store the frame for this client
        with lock:
            client_frames[sid] = frame
            
            # Also update global frame buffer for compatibility with existing code
            global frame_buffer
            frame_buffer = frame
            
    except Exception as e:
        print(f"Error processing client frame: {e}")

@sio.event
def request_processed_frame(sid, data=None):
    """Client requests processed frame with face recognition"""
    try:
        # Send processed frame if available
        if sid in client_processed_frames and client_processed_frames[sid] is not None:
            processed_frame = client_processed_frames[sid]
            
            # Convert frame to base64
            success, buffer = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if success:
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                sio.emit('processed_frame', {'image': f'data:image/jpeg;base64,{image_base64}'}, room=sid)
    except Exception as e:
        print(f"Error sending processed frame: {e}")

# Flask routes
@flask_app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@flask_app.route('/analyze_current_frame', methods=['POST'])
def analyze_current_frame():
    """Analyze the current frame with GPT-4 Vision"""
    global last_processed_time, is_speaking
    
    current_time = time.time()
    
    # Rate limiting to prevent too frequent API calls
    if current_time - last_processed_time < processing_interval:
        return jsonify({"error": "Please wait before sending another request"}), 429
    
    # Get client's session ID from request
    sid = request.sid if hasattr(request, 'sid') else None
    
    with lock:
        # Use client-specific frame if available
        if sid and sid in client_frames and client_frames[sid] is not None:
            frame_to_analyze = client_frames[sid].copy()
        elif default_client_sid and default_client_sid in client_frames and client_frames[default_client_sid] is not None:
            frame_to_analyze = client_frames[default_client_sid].copy()
        elif frame_buffer is not None:
            # Fall back to global frame buffer
            frame_to_analyze = frame_buffer.copy()
        else:
            return jsonify({"error": "No frame available"}), 404
    
    # Update the last processed time
    last_processed_time = current_time
    
    # Pause listening while processing
    pause_listening()
    is_speaking = True
    sio.emit('listening_status', {'is_listening': False, 'reason': 'processing'})
    
    # Submit analysis to thread pool
    analysis_future = api_thread_pool.submit(analyze_frame_with_gpt, frame_to_analyze, "What can you see in this image?")
    analysis = analysis_future.result()
    
    # Generate speech with OpenAI's TTS (also through thread pool)
    tts_future = api_thread_pool.submit(text_to_speech_openai, analysis)
    tts_file = tts_future.result()
    
    return jsonify({
        "analysis": analysis,
        "audio_file": tts_file
    })

@flask_app.route('/toggle_listening', methods=['POST'])
def toggle_listening_route():
    """Toggle between passive and active listening modes"""
    global is_listening, is_passive_listening
    
    # If active listening is on
    if is_listening:
        # Switch to passive
        stop_active_return_to_passive()
        mode = "passive"
    elif is_passive_listening:
        # If passive is on, switch to active
        stop_passive_listening()
        start_listening()
        mode = "active"
    else:
        # If nothing is on, start with passive
        start_passive_listening()
        mode = "passive" 
    
    return jsonify({
        "success": True,
        "is_listening": is_listening,
        "is_passive_listening": is_passive_listening,
        "mode": mode
    })

@flask_app.route('/get_faces', methods=['GET'])
def get_faces():
    """Get list of recognized faces"""
    try:
        return jsonify({
            "faces": face_recognizer.known_face_names
        })
    except Exception as e:
        print(f"Error getting faces: {e}")
        return jsonify({"error": str(e)}), 500

@flask_app.route('/refresh_face_database', methods=['POST'])
def refresh_face_database():
    """Manually refresh the face database"""
    try:
        face_recognizer.build_face_database()
        return jsonify({
            "success": True,
            "faces": face_recognizer.known_face_names
        })
    except Exception as e:
        print(f"Error refreshing face database: {e}")
        return jsonify({"error": str(e)}), 500

@flask_app.route('/get_audio_queue', methods=['GET'])
def get_audio_queue():
    """Get audio processing results from queue (used for polling)"""
    try:
        # Non-blocking queue get with timeout
        data = audio_queue.get(block=False)
        return jsonify(data)
    except queue.Empty:
        return jsonify({"type": "empty"})
    except Exception as e:
        return jsonify({"type": "error", "message": str(e)})

@flask_app.route('/static/audio/<filename>')
def serve_audio(filename):
    """Serve the generated audio files"""
    return send_file(f"static/audio/{filename}")

@flask_app.route('/listening_status', methods=['GET'])
def get_listening_status():
    """Get the current listening status"""
    if is_speaking:
        status = "speaking"
    elif is_listening:
        status = "active"
    elif is_passive_listening:
        status = "passive"
    else:
        status = "inactive"
        
    return jsonify({
        "is_listening": is_listening,
        "is_passive_listening": is_passive_listening,
        "is_speaking": is_speaking,
        "status": status
    })

@flask_app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up resources before closing"""
    try:
        # Stop listening if active
        if is_listening:
            stop_listening()
        
        # Stop passive listening if active
        if is_passive_listening:
            stop_passive_listening()
        
        # Shutdown thread pools
        thread_pool.shutdown(wait=False)
        api_thread_pool.shutdown(wait=False)
        
        # Clear all client frames
        with lock:
            client_frames.clear()
            client_processed_frames.clear()
            
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs("static/audio", exist_ok=True)
    os.makedirs("static/faces", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Start in passive listening mode by default
    print("Starting in passive listening mode")
    start_passive_listening()
    
    # Use a regular WSGI server instead of Flask's development server
    from waitress import serve
    print(f"Starting server on http://127.0.0.1:5000 - {AI_NAME} is ready to help!")
    serve(app, host="127.0.0.1", port=5000, threads=8)