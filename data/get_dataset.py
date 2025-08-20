import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
import os
import csv
from dotenv import load_dotenv
import string
import pylast
import json
import subprocess
import pandas as pd

def get_playlist_tracks(sp, n_tracks, cluster_size):
    print(f"Getting playlist tracks for {dataset_name}")
    queries = []
    with open("playlist_queries.txt", "r") as f:
        queries = f.readlines()

    with open(dataset_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        track_count = 0
        playlist_count = 0
        while track_count < n_tracks:
            query = random.choice(queries) #get random query to search for playlist
            results = sp.search(q=query, type="playlist", limit=1, offset=random.randint(0, 950))
            try:
                results['playlists']['items'][0]
            except:
                continue    
            if results['playlists']['items'][0]:
                playlist = results['playlists']['items'][0]
                playlist_count += 1
                tracks = sp.playlist_tracks(playlist['id'])
                if len(tracks['items']) > 5:
                    track_indices = random.sample(range(len(tracks['items'])), 5) #get random track indices from the playlist
                    for cluster_count, i in enumerate(track_indices):
                        track = tracks['items'][i]
                        if track and track['track']:
                            if test_set:
                                if track['track']['id'] in train_set_ids:
                                    continue
                            writer.writerow([track['track']['name'], track['track']['artists'][0]['name'], track['track']['id'], f'p{playlist_count}', 'playlist', track['track']['preview_url']]) #write random track info to csv
                            track_count += 1
                            if cluster_count >= cluster_size:
                                break

def get_album_tracks(sp, n_tracks, cluster_size):
    print(f"Getting album tracks for {dataset_name}")
    album_count = 0
    track_count = 0
    with open(dataset_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        while track_count < n_tracks:
            #get random letter for query
            letter = random.choice(string.ascii_lowercase)
            #get random album from letter
            results = sp.search(q=letter, type="album", limit=1, market="US", offset=random.randint(0, 100)) #way lower offset here so we dont get too foreign albums (or just use market=US)
            if results['albums']['items'][0]:
                album = results['albums']['items'][0]
                album_count += 1
                tracks = sp.album_tracks(album['id'])
                if len(tracks['items']) > 5:
                    track_indices = random.sample(range(len(tracks['items'])), 5)
                    for cluster_count, i in enumerate(track_indices):
                        track = tracks['items'][i]
                        if track:
                            writer.writerow([track['name'], track['artists'][0]['name'], track['id'], f'a{album_count}', 'album', track['preview_url']])
                            track_count += 1
                            if cluster_count >= cluster_size:
                                break
    
def get_genre_tracks(sp):
    genres = sp.recommendation_genre_seeds()['genres'] #this api is deprecated, so not sure if this is a possible source
    print(genres)

def get_lastfm_tracks(sp, n_tracks, cluster_size):
    print(f"Getting lastfm tracks for {dataset_name}")
    network = pylast.LastFMNetwork(
        api_key=os.getenv("LASTFM_API_KEY"),
        api_secret=os.getenv("LASTFM_API_SECRET"),
        username=os.getenv("LASTFM_USERNAME"),
        password_hash=pylast.md5(os.getenv("LASTFM_PASSWORD")))
    
    #get global top 1000 tracks
    top_tracks = network.get_top_tracks(limit=1000) #max is 1000 for this api
    track_count = 0
    lastfm_count = 0
    with open(dataset_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        while track_count < n_tracks:
            lastfm_count += 1
            tracks_to_write = []
            #get random track from top 1000
            #track = random.choice(top_tracks)
            track = top_tracks[lastfm_count%1000]
            tracks_to_write.append(track)
            #get cluster_size-1 top similar tracks to query
            similar_tracks = track.item.get_similar(limit=cluster_size-1)
            tracks_to_write.extend(similar_tracks)
            #write tracks to csv
            for cluster_count, track in enumerate(tracks_to_write):
                #find track on spotify
                sp_track = sp.search(q=f"{track.item.get_name()} {track.item.get_artist().get_name()}", type="track", limit=1)
                if sp_track['tracks']['items'][0]:
                    if test_set:
                        if sp_track['tracks']['items'][0]['id'] in train_set_ids:
                            continue
                    writer.writerow([sp_track['tracks']['items'][0]['name'], sp_track['tracks']['items'][0]['artists'][0]['name'], sp_track['tracks']['items'][0]['id'], f'l{lastfm_count}', 'lastfm', sp_track['tracks']['items'][0]['preview_url']])
                track_count += 1
                if cluster_count >= cluster_size:
                    break


def get_preview_urls():
    print("Getting preview urls...")
    with open(dataset_name, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        tracks_json = []
        next(reader)  # Skip header row
        for row in reader:
            tracks_json.append({
                "name": row[0],
                "artist": row[1],
                "spotify_track_id": row[2]
            })
    
    with open("tracks.json", 'w', encoding='utf-8') as f:
        json.dump(tracks_json, f, indent=2)

    node_cmd = ["node", "preview_finder.js"]
    env = os.environ.copy()
    print('running preview finder')
    proc = subprocess.run(node_cmd, capture_output=True, text=True, env=env, cwd="node")
    print(proc.stdout)

    if proc.returncode != 0:
        print(f"Error: {proc.stderr}")
        return

    #read preview_urls.json from preview_finder.js
    with open("preview_urls.json", 'r', encoding='utf-8') as f:
        preview_urls = json.load(f)

    print(preview_urls)
    #write preview_urls to csv
    print('writing preview urls to csv')
    df = pd.read_csv(dataset_name)
    df["previewURL"] = df["trackID"].map(preview_urls)
    df.to_csv(dataset_name, index=False)
    
def main():

    load_dotenv()

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")))

    # Set up CSV file with headers
    

    #get train set
    global test_set 
    global dataset_name
    dataset_name = "train_tracks.csv"
    test_set = False

    with open(dataset_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'artist', 'trackID', 'clusterID', 'clusterType', 'previewURL'])

    get_playlist_tracks(sp, n_tracks=75000, cluster_size=5)
    get_album_tracks(sp, n_tracks=75000, cluster_size=5)
    #get_lastfm_tracks(sp, n_tracks=5000, cluster_size=5)
    get_preview_urls()

    #get test set
    test_set = True
    train_set = pd.read_csv(dataset_name)
    dataset_name = "test_tracks.csv"
    #get list of spotify track ids in train set
    global train_set_ids
    train_set_ids = train_set["trackID"].tolist()

    with open(dataset_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'artist', 'trackID', 'clusterID', 'clusterType', 'previewURL'])
    #get_playlist_tracks(sp, n_tracks=150, cluster_size=15)
    #get_album_tracks(sp, n_tracks=150, cluster_size=15)
    get_lastfm_tracks(sp, n_tracks=300, cluster_size=15)
    get_preview_urls()


if __name__ == "__main__":
    main()
