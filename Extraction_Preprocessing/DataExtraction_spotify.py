import spotipy
from spotipy.oauth2 import SpotifyClientCredentials 


client_id = "56f8b5336d824238855bf2046b43f77f"
client_secret = "ef5b5b1e55844c5eae3e98cc8d0b9989"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


#function for making lists of artist b=names and their spotify uris
#this function only gives the names of albums and uris for a perticular artist
def alb_uri(name):
    result = sp.search(name)
    
    art_uri = result['tracks']['items'][1]['artists'][0]['uri'] 
    
    #results contain dictionary returned by the sp.search function od spotipy
    
    
    #
    artist_album = sp.artist_albums(art_uri)
    
    
    artist_album_names = []
    artist_album_uris = []
    
    cnt = 0
    #for each item in the 
    for i in range(len(artist_album['items'])):
        artist_album_names.append(artist_album['items'][i]['name'])
        artist_album_uris.append(artist_album['items'][i]['uri'])
        cnt += 1
        
        if cnt == 5:
            break
    return artist_album_names,artist_album_uris

def album_songs(uri):
    album = uri 
    spotify_albums[album] = {}
    #Create keys-values of empty lists inside nested dictionary for album
    spotify_albums[album]['album'] = [] 
    spotify_albums[album]['track_number'] = []
    spotify_albums[album]['id'] = []
    spotify_albums[album]['name'] = []
    spotify_albums[album]['uri'] = []
    #pull data on album tracks
    
    tracks = sp.album_tracks(album)

        
    cnt = 0
    for n in range(len(tracks['items'])):
        
        
        spotify_albums[album]['album'].append(artist_album_names[album_count]) 
        spotify_albums[album]['track_number'].append(tracks['items'][n]['track_number'])
        spotify_albums[album]['id'].append(tracks['items'][n]['id'])
        spotify_albums[album]['name'].append(tracks['items'][n]['name'])
        spotify_albums[album]['uri'].append(tracks['items'][n]['uri'])
        cnt +=1
        if cnt == 4:
            break
        
        
def audio_features(album):
    #Add new key-values to store audio features
    spotify_albums[album]['acousticness'] = []
    spotify_albums[album]['danceability'] = []
    spotify_albums[album]['energy'] = []
    spotify_albums[album]['instrumentalness'] = []
    spotify_albums[album]['liveness'] = []
    spotify_albums[album]['loudness'] = []
    spotify_albums[album]['speechiness'] = []
    spotify_albums[album]['tempo'] = []
    spotify_albums[album]['valence'] = []
    spotify_albums[album]['popularity'] = []
    
    track_count = 0
    for track in spotify_albums[album]['uri']:
        #pull audio features per track
        features = sp.audio_features(track)
        
        #Append to relevant key-value
        
        try:
            spotify_albums[album]['acousticness'].append(features[0]['acousticness'])
            spotify_albums[album]['danceability'].append(features[0]['danceability'])
            spotify_albums[album]['energy'].append(features[0]['energy'])
            spotify_albums[album]['instrumentalness'].append(features[0]['instrumentalness'])
            spotify_albums[album]['liveness'].append(features[0]['liveness'])
            spotify_albums[album]['loudness'].append(features[0]['loudness'])
            spotify_albums[album]['speechiness'].append(features[0]['speechiness'])
            spotify_albums[album]['tempo'].append(features[0]['tempo'])
            spotify_albums[album]['valence'].append(features[0]['valence'])
            #popularity is stored elsewhere
            pop = sp.track(track)
            spotify_albums[album]['popularity'].append(pop['popularity'])
            track_count+=1
        except Exception:
            continue
        
        

indian_music_artists =  set([   "Pink", "Bon Jovi", "Beyoncé", "Fleetwood Mac", "Guns N' Roses", "AC/DC", "The Eagles", "Whitney Houston", "Phil Collins", "Nirvana", "Bruce Springsteen", "Britney Spears", "Eminem", "Shakira", "Elton John", "Justin Bieber", "Bon Jovi", "Jennifer Lopez", "Kanye West", "U2", "Justin Timberlake", "Eric Clapton"])
#making lists of the albums and their uris


artist_album_names = []
artist_album_uris = []
    
for i in indian_music_artists:  #for each artist in the list adding the album names and their uris in the names and uri lists
    name  , uri = alb_uri(i)
    artist_album_names = artist_album_names + name
    artist_album_uris = artist_album_uris+uri
    
    


#####empty dictionary fo storing the album 
spotify_albums = {} 
album_count = 0
for uri in artist_album_uris: #each album
    album_songs(uri)
    print(str(artist_album_names[album_count]) + " album songs has been added to spotify_albums dictionary")
    album_count+=1 #Updates album count once all tracks have been added
    


import time
import numpy as np
sleep_min = 2
sleep_max = 5
start_time = time.time()
request_count = 0
for i in spotify_albums:
    try:
        audio_features(i)
    except:
        continue
    request_count+=1
    if request_count % 5 == 0:
        print(str(request_count) + " playlists completed")
        time.sleep(np.random.uniform(sleep_min, sleep_max))
        print('Loop #: {}'.format(request_count))
        print('Elapsed Time: {} seconds'.format(time.time() - start_time))
        
        
        
dic_df = {}
dic_df['album'] = []
dic_df['track_number'] = []
dic_df['id'] = []
dic_df['name'] = []
dic_df['uri'] = []
dic_df['acousticness'] = []
dic_df['danceability'] = []
dic_df['energy'] = []
dic_df['instrumentalness'] = []
dic_df['liveness'] = []
dic_df['loudness'] = []
dic_df['speechiness'] = []
dic_df['tempo'] = []
dic_df['valence'] = []
dic_df['popularity'] = []
for album in spotify_albums: 
    for feature in spotify_albums[album]:
        dic_df[feature].extend(spotify_albums[album][feature])
        
len(dic_df['album'])


import pandas as pd
dataframe = pd.DataFrame.from_dict(dic_df)
dataframe    


print(len(dataframe))
final_df = dataframe.sort_values('popularity', ascending=False).drop_duplicates('name').sort_index()
print(len(final_df))
        
dataframe.to_csv("spotify_music6.csv")

############################
for i in dic_df.keys():
    print(i,len(dic_df[i]))
        
    





        
        
        
        
        
        
        
        
        
        
        
        