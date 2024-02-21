
import pandas as pd
import pymongo
from pymongo import MongoClient
import numpy as np

import cv2
from PIL import Image

from sklearn.preprocessing import MinMaxScaler
from skimage import io
import matplotlib.pyplot as plt


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import requests
import os

import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
from keras.models import load_model


client = MongoClient('mongodb://localhost:27017/')  # Update the connection string as needed
db = client['songs']  # Replace 'your_database' with the database name
collection = db['all_songs']  # Replace 'your_collection' with the collection name

# Example MongoDB Query
# Replace this query with your specific MongoDB query to filter the data as needed
# In this example, it retrieves all documents from the collection
query = {}

# Fetch data from MongoDB based on the query
cursor = collection.find(query)

# Iterate over the cursor to process each document
for document in cursor:
    # Extract relevant fields from the document
    track_name = document['track_name']
    artist_name = document['artist_name']
    track_uri = document['track_uri']
    # Process the data as needed in your code
   # print(f"Track: {track_name}, Artist: {artist_name}, URI: {track_uri}")

# Close MongoDB connection
client.close()

scope = 'user-library-read'
client_id = '3a98f83d33c547668bcd83dbd32eb699'
client_secret = '93f3ad98a52545d0884323e1ad45344a'    

# Here In this Line creating instance of SpotifyClientCredentials with the provided client_id and client_secret

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

# Creating a Spotify instance (sp) with the created auth_manager

sp = spotipy.Spotify(auth_manager=auth_manager)


# Prompting the user to authenticate and authorize the application through Spotify
# This line initiates the OAuth 2.0 authorization flow to obtain a user-specific access token
token = util.prompt_for_user_token(scope, client_id=client_id, client_secret=client_secret, redirect_uri='http://localhost:8400/')


# Creating a Spotify API client object using the obtained access token

sp = spotipy.Spotify(auth=token)

#gather playlist names from user account playlists
#This Id name use as the parameter of Get_Spotify_Playlist_DataFrame function which is called in Recommend_top1function 
id_name = {}
for i in sp.current_user_playlists()['items']:
    id_name[i['name']] = i['uri'].split(':')[2]
#print(id_name)

# This function filters the Spotify DataFrame based on the given emotion (x) to create a specific dataset.
# The resulting dataset includes songs associated with the specified moods related to the given emotion.
def ChooseDataset(x, collection):
    if x == "Disgust":
        return collection.find({'Mood': {'$in': ['Sad', 'Calm']}})
    if x == "Angry":
        return collection.find({'Mood': {'$in': ['Energetic', 'Sad']}})
    if x == "Fear":
        return collection.find({'Mood': {'$in': ['Energetic', 'Calm', 'Sad']}})
    if x == "Happy":
        return collection.find({'Mood': {'$in': ['Energetic', 'Happy']}})
    if x == "Sad":
        return collection.find({'Mood': 'Sad'})
    if x == "Surprise":
        return collection.find({'Mood': {'$in': ['Energetic', 'Happy']}})
    if x == "Neutral":
        return collection.find({'Mood': {'$in': ['Happy', 'Calm']}})
    return collection.find() 

loaded_model = load_model('D:\\Music-recommendation-based-on-facial-emotion-recognition-main (1)\\Music-recommendation-based-on-facial-emotion-recognition-main\\new_img_data\\model.keras', compile = True)

# Function to recommend songs
def Recommend_Playlist(collection):
    # Randomly sample 10 songs from the provided MongoDB collection
    recommended_songs = list(collection.aggregate([{'$sample': {'size': 10}}]))
    
    return recommended_songs


#This function retrieves the album cover image URL for a given Spotify track using the Spotify API. 
def get_spotify_track_image(track_id, spotify_token):
    # Set up headers with the provided Spotify API access token
    headers = {
        'Authorization': f'Bearer {spotify_token}',
    }

    # Spotify API endpoint to get track details
    endpoint = f'https://api.spotify.com/v1/tracks/{track_id}'
    
    # Make a GET request to the Spotify API
    response = requests.get(endpoint, headers=headers)
    
    if response.status_code == 200:
        # Extract the image URL from the API response
        try:
            image_url = response.json()['album']['images'][0]['url']
            return image_url
        except IndexError:
            print(f"Error: No images found for track {track_id}")
            return None
    else:
        # If the request fails, print the response and return None
        print(f"Error: {response.status_code} - {response.text}")
        return None
def visualize_songs(top10_recommendation, spotify_token):
    # Setting up the plot figure with appropriate size
    plt.figure(figsize=(15, int(0.625 * len(top10_recommendation))))
    
    # Defining the number of columns in the grid
    columns = 3
    
    # Loop through each track and display its album cover image
    for i, track in enumerate(top10_recommendation):
        track_id = track['id']
        
        # Get the image URL for the track from Spotify API
        image_url = get_spotify_track_image(track_id, spotify_token)
        
        if image_url:
            # Set up a subplot for each track in the grid
            plt.subplot(int(len(top10_recommendation) / columns) + 1, columns, i + 1)
            
            try:
                # Attempt to load the image; handle cases where the image is truncated or invalid.
                image_content = requests.get(image_url).content
                image = io.imread(image_content, plugin='imageio')
                
                # Display the image and track information.
                plt.imshow(image)
                plt.xticks(color='b', fontsize=0.1)
                plt.yticks(color='b', fontsize=0.1)
                plt.xlabel(track['track_name'], fontsize=12)
                plt.tight_layout(h_pad=0.4, w_pad=0)
                plt.subplots_adjust(wspace=None, hspace=None)
            
            except Exception as e:
                print(f"Error loading image for track {track_id}: {e}")

    plt.show()
# Main function for recommendation
def Recommend_Top10(emotions, collection):
    # Choose the dataset based on the emotion
    emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    emotion = emotion_dict[emotions[-1]]  # Use the most recent emotion

    O_df = ChooseDataset(emotion,collection)

    # Recommended Songs stored here
    top10_recommendation = Recommend_Playlist(collection)
    print("Detected Emotion is :- ",emotion)
    
    # Visualize recommended songs
    visualize_songs(top10_recommendation,token)


def moodNamePrintFromLabel(n):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[n]

# Function to capture video from webcam and perform emotion detection
def capture_and_detect_emotion(model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    video_cap = cv2.VideoCapture(0)  # Open the webcam

    emotions = []
    start_time = cv2.getTickCount()

    while (video_cap.isOpened()):
        ret, frame = video_cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

        for (x, y, w, h) in faces:
            pxl_lst = []
            for i in range(y, y + h):
                lst = []
                for j in range(x, x + w):
                    lst.append(gray_img[i][j])
                pxl_lst.append(lst)
            single_face = np.array(pxl_lst)

            resized_img = cv2.resize(single_face, (48, 48), interpolation=cv2.INTER_AREA)
            resized_img = np.reshape(resized_img, (1, 48, 48, 1)) / 255.0

            # Get the result from the model
            result = np.argmax(model.predict(resized_img), axis=-1)

            # Draw a rectangle around the face and display the mood label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            cv2.putText(frame, moodNamePrintFromLabel(result[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            
            emotions.append(result[0])

        # Display the frame with mood information
        cv2.imshow('Webcam Mood Detection', frame)

        current_time = cv2.getTickCount()
        elapsed_time = (current_time - start_time) / cv2.getTickFrequency()

        if elapsed_time >= 5:
            break  # Break after 5 seconds

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    video_cap.release()
    cv2.destroyAllWindows()

    return emotions


client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["songs"]
collection = db["all_songs"]

# Capture emotions from webcam
emotions = capture_and_detect_emotion(loaded_model)

# Recommend songs based on the most recent emotion from MongoDB collection
Recommend_Top10(emotions, collection)
