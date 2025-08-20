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
import time
import numpy as np


def get_playlist_tracks(sp, n_tracks=None, cluster_size=4):
    """
    Gets 1000 playlists from each query and cluster_size tracks from each playlist
    Max n_tracks is cluster_size*950*len(queries (currently 25))
    """
    
    print(f"Getting playlist tracks for {dataset_name}")
    queries = []
    with open("playlist_queries.txt", "r") as f:
        queries = f.readlines()

    if n_tracks is None:
        n_tracks = cluster_size*950*len(queries) #set default to max n_tracks

    with open(dataset_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        track_count = 0
        playlist_count = 0
        i = 0

        while track_count < n_tracks:
            try:
                query = queries[i]
            except IndexError:
                print('Exausted playlist queries')
                return
            i+=1

            #results = sp.search(q=query, type="playlist", limit=1, offset=random.randint(0, 950))
            results = []
            for i in range(0, min(950, n_tracks), 50):
                results.extend(sp.search(q=query, type="playlist", limit=50, offset=i)['playlists']['items'])
                time.sleep(0.5)
            print(f'Found {len(results)} playlists for {query}')
  
            for playlist in results:
                if playlist:
                    playlist_count += 1
                    tracks = sp.playlist_tracks(playlist['id'])
                    time.sleep(0.2)
                    if len(tracks['items']) > 5:
                        track_indices = random.sample(range(len(tracks['items'])), 5) #get random track indices from the playlist
                        for cluster_count, i in enumerate(track_indices):
                            track = tracks['items'][i]
                            if track and track['track']:
                                #if test_set:
                                    #if track['track']['id'] in train_set_ids:
                                        #continue
                                writer.writerow([track['track']['name'], track['track']['artists'][0]['name'], track['track']['id'], f'p{playlist_count}', 'playlist', track['track']['preview_url']]) #write random track info to csv
                                track_count += 1
                                if track_count%100 == 0:
                                    print(f"Playlist tracks: {track_count}", end="\r")
                                    csvfile.flush()
                                    time.sleep(0.2)
                                if cluster_count >= cluster_size:
                                    break

def get_album_tracks(sp, n_tracks=None, cluster_size=4):
    """
    Gets 1000 albums from each letter and cluster_size tracks from each album
    Max n_tracks is cluster_size*500*26
    """
    print(f"Getting album tracks for {dataset_name}")

    if n_tracks is None:
        n_tracks = cluster_size*500*26 #set default to max n_tracks

    album_count = 0
    track_count = 0
    i=0
    with open(dataset_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        while track_count < n_tracks:
            #get random letter for query
            try:
                letter = string.ascii_lowercase[i]
            except IndexError:
                print('Exausted album queries')
                return
            i+=1
            #get random album from letter
            results = []
            for i in range(0, min(500, n_tracks), 50):
                results.extend(sp.search(q=letter, type="album", limit=50, offset=i)['albums']['items'])
                time.sleep(0.5)
            print(f'Found {len(results)} albums for {letter}')

            for album in results:
                if album:
                    album_count += 1
                    tracks = sp.album_tracks(album['id'])
                    time.sleep(0.2)
                    if len(tracks['items']) > cluster_size:
                        track_indices = random.sample(range(len(tracks['items'])), cluster_size)
                        for cluster_count, i in enumerate(track_indices):
                            track = tracks['items'][i]
                            if track:
                                writer.writerow([track['name'], track['artists'][0]['name'], track['id'], f'a{album_count}', 'album', track['preview_url']])
                                track_count += 1
                                if track_count%100 == 0:
                                    print(f"Album tracks: {track_count}", end="\r")
                                    csvfile.flush()
                                    time.sleep(0.2)
                                if cluster_count >= cluster_size:
                                    break
        

def get_lastfm_tracks(sp, n_tracks=None, cluster_size=4):
    """
    Gets seed tracks from lastfm dataset (76,036 tracks) and cluster_size similar tracks from each seed track using Last.fm similarity API
    Max n_tracks is cluster_size*76,036
    """
    print(f"Getting lastfm tracks for {dataset_name}")
    network = pylast.LastFMNetwork(
        api_key=os.getenv("LASTFM_API_KEY"),
        api_secret=os.getenv("LASTFM_API_SECRET"),
        username=os.getenv("LASTFM_USERNAME"),
        password_hash=pylast.md5(os.getenv("LASTFM_PASSWORD")))

    if n_tracks is None:
        n_tracks = cluster_size*76036 #set default to max n_tracks
    
    #top_tracks = network.get_top_tracks(limit=1000) #max is 1000 for this api
    lastfm_tracks = pd.read_csv('Lastfm_reduced.csv')
    track_count = 0
    lastfm_count = 0
    with open(dataset_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        while track_count < n_tracks:
            lastfm_count += 1
            tracks_to_write = []
            #get random track from top 1000
            #track = random.choice(top_tracks)
            #track = top_tracks[lastfm_count%1000]
            #get artist and track from the CSV row
            try:
                row = lastfm_tracks.iloc[lastfm_count]
            except IndexError:
                print('Exausted lastfm dataset')
                return
            artist_name = row['Artist']
            track_name = row['Track']
            #search for the track using Last.fm API
            track = network.get_track(artist_name, track_name)
            tracks_to_write.append(track)
            
            #get cluster_size-1 top similar tracks to query
            similar_tracks = track.get_similar(limit=cluster_size-1)
            tracks_to_write.extend(similar_tracks)
            #write tracks to csv
            for cluster_count, track in enumerate(tracks_to_write):
                #find track on spotify
                try:
                    track = track.item
                except:
                    pass
                time.sleep(0.2)
                sp_track = sp.search(q=f"{track.get_name()} {track.get_artist().get_name()}", type="track", limit=1)
                if sp_track['tracks']['items'][0]:
                    #if test_set:
                        #if sp_track['tracks']['items'][0]['id'] in train_set_ids:
                            #continue
                    writer.writerow([sp_track['tracks']['items'][0]['name'], sp_track['tracks']['items'][0]['artists'][0]['name'], sp_track['tracks']['items'][0]['id'], f'l{lastfm_count}', 'lastfm', sp_track['tracks']['items'][0]['preview_url']])
                track_count += 1
                if track_count%100 == 0:
                    print(f"Lastfm tracks: {track_count}", end="\r")
                    csvfile.flush()
                    time.sleep(0.2)
                if cluster_count >= cluster_size:
                    break


def get_preview_urls(n_processes=10):
    print("Getting preview urls...")

    full_df = pd.read_csv(dataset_name)
    #split intto n_processes datasets
    df_list = np.array_split(full_df, n_processes)
    for i, df in enumerate(df_list):
        tracks_json = []
        for index, row in df.iterrows():
            tracks_json.append({
                    "name": row[0],
                    "artist": row[1],
                    "spotify_track_id": row[2]
            })
    
        with open(f"json/tracks_{i}.json", 'w', encoding='utf-8') as f:
            json.dump(tracks_json, f, indent=2)
    
    processes = []
    for i in range(n_processes):
        node_cmd = ["node", "preview_finder.js"]
        env = os.environ.copy()
        env["TRACKS_IDX"] = str(i)
        print(f'running preview finder for subprocess {i}')
        proc = subprocess.Popen(node_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd="node")
        time.sleep(1)
        processes.append(proc)
    
    # Wait for all processes to complete
    for i, proc in enumerate(processes):
        stdout, stderr = proc.communicate()
        print(f'Subprocess {i} output:')
        #print(stdout)
        if proc.returncode != 0:
            print(f"Error in subprocess {i}: {stderr}")

    #read preview_urls.json from preview_finder.js
    preview_urls = {}
    for i in range(n_processes):
        with open(f"json/preview_urls_{i}.json", 'r', encoding='utf-8') as f:
            preview_urls.update(json.load(f))

    #write preview_urls to csv
    print('writing preview urls to csv')
    df = pd.read_csv(dataset_name)
    df["previewURL"] = df["trackID"].map(preview_urls)
    df.to_csv(dataset_name, index=False)
    
def main():

    load_dotenv()

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")),
        retries=10,
        )

    #get train set
    #global test_set 
    global dataset_name
    dataset_name = "train_tracks.csv"
    #test_set = False

    with open(dataset_name, 'w', newline='', encoding='utf-8') as csvfile: #set up csv file with headers
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'artist', 'trackID', 'clusterID', 'clusterType', 'previewURL'])

    get_playlist_tracks(sp, n_tracks=8, cluster_size=4) #default 95,000 tracks
    get_album_tracks(sp, n_tracks=8, cluster_size=4) #default 52,000 tracks
    get_lastfm_tracks(sp, n_tracks=8, cluster_size=4) #default 304,144 tracks
    get_preview_urls(n_processes=10)

    #get test set
    """test_set = True
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
    #get_preview_urls()"""


if __name__ == "__main__":
    main()
