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
from scripts.spotify_preview import get_spotify_preview_url


def get_playlist_tracks(sp, n_tracks=None, cluster_size=4, start_query=0):
    """
    Gets 1000 playlists from each query and cluster_size tracks from each playlist
    Max n_tracks is cluster_size*950*len(queries (currently 25))
    """
    
    print(f"Getting playlist tracks for {dataset_name}")
    queries = []
    with open("playlist_queries.txt", "r") as f:
        queries = f.readlines()

    if n_tracks is None:
        n_tracks = cluster_size*200*len(queries) #set default to max n_tracks

    with open(dataset_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        track_count = 0
        playlist_count = 0
        query_count = start_query

        while track_count < n_tracks:
            try:
                query = queries[query_count]
            except IndexError:
                print('Exausted playlist queries')
                return
            query_count+=1
            with open('.resume', 'w') as f:
                f.write(f'p:{query_count}')
            

            #results = sp.search(q=query, type="playlist", limit=1, offset=random.randint(0, 950))
            results = []
            for i in range(0, min(200, n_tracks), 50):
                results.extend(sp.search(q=query, type="playlist", limit=50, offset=i)['playlists']['items'])
                time.sleep(2)
            print(f'Found {len(results)} playlists for {query}')
  
            for playlist in results:
                if playlist:
                    playlist_count += 1
                    tracks = sp.playlist_tracks(playlist['id'])
                    time.sleep(0.7)
                    if len(tracks['items']) > 5:
                        track_indices = random.sample(range(len(tracks['items'])), 5) #get random track indices from the playlist
                        for cluster_count, j in enumerate(track_indices):
                            track = tracks['items'][j]
                            if track and track['track']:
                                #if test_set:
                                    #if track['track']['id'] in train_set_ids:
                                        #continue
                                preview_url = get_spotify_preview_url(track['track']['id'])
                                if preview_url:
                                    writer.writerow([track['track']['name'], track['track']['artists'][0]['name'], track['track']['id'], f'p{playlist_count}', 'playlist', preview_url]) #write random track info to csv
                                    track_count += 1
                                if track_count%100 == 0:
                                    print(f"Playlist tracks: {track_count}", end="\r")
                                    csvfile.flush()
                                    time.sleep(0.2)
                                    #get_preview_urls()
                                if cluster_count >= cluster_size:
                                    break

def get_album_tracks(sp, n_tracks=None, cluster_size=4, start_query=0):
    """
    Gets 1000 albums from each letter and cluster_size tracks from each album
    Max n_tracks is cluster_size*500*26
    """
    print(f"Getting album tracks for {dataset_name}")

    if n_tracks is None:
        n_tracks = cluster_size*100*26 #set default to max n_tracks

    album_count = 0
    track_count = 0
    query_counter = start_query
    with open(dataset_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        while track_count < n_tracks:
            #get letter for query
            try:
                letter = string.ascii_lowercase[query_counter]
            except IndexError:
                print('Exausted album queries')
                return

            query_counter+=1
            with open('.resume', 'w') as f:
                f.write(f'a:{query_counter}')
            #get random album from letter
            results = []
            for i in range(0, min(100, n_tracks), 50):
                results.extend(sp.search(q=letter, type="album", limit=50, offset=i)['albums']['items'])
                time.sleep(1)
            print(f'Found {len(results)} albums for {letter}')

            for album in results:
                if album:
                    album_count += 1
                    tracks = sp.album_tracks(album['id'])
                    time.sleep(0.7)
                    if len(tracks['items']) > cluster_size:
                        track_indices = random.sample(range(len(tracks['items'])), cluster_size)
                        for cluster_count, j in enumerate(track_indices):
                            track = tracks['items'][j]
                            if track:
                                preview_url = get_spotify_preview_url(track['id'])
                                if preview_url:
                                    writer.writerow([track['name'], track['artists'][0]['name'], track['id'], f'a{album_count}', 'album', preview_url])
                                    track_count += 1
                                if track_count%100 == 0:
                                    print(f"Album tracks: {track_count}", end="\r")
                                    csvfile.flush()
                                    time.sleep(0.2)
                                    #get_preview_urls()
                                if cluster_count >= cluster_size:
                                    break
        

def get_lastfm_tracks(sp, n_tracks=None, cluster_size=4, start_query=0):
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
    query_counter = start_query
    with open(dataset_name, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        while track_count < n_tracks:
            tracks_to_write = []
            #get random track from top 1000
            #track = random.choice(top_tracks)
            #track = top_tracks[lastfm_count%1000]
            #get artist and track from the CSV row
            try:
                row = lastfm_tracks.iloc[query_counter]
            except IndexError:
                print('Exausted lastfm dataset')
                return

            query_counter+=1
            with open('.resume', 'w') as f:
                f.write(f'l:{query_counter}')

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
                time.sleep(0.7)
                sp_track = sp.search(q=f"{track.get_name()} {track.get_artist().get_name()}", type="track", limit=1)
                if sp_track['tracks']['items'][0]:
                    #if test_set:
                        #if sp_track['tracks']['items'][0]['id'] in train_set_ids:
                            #continue
                    preview_url = get_spotify_preview_url(sp_track['tracks']['items'][0]['id'])
                    if preview_url:
                        writer.writerow([sp_track['tracks']['items'][0]['name'], sp_track['tracks']['items'][0]['artists'][0]['name'], sp_track['tracks']['items'][0]['id'], f'l{query_counter}', 'lastfm', preview_url])
                track_count += 1
                if track_count%100 == 0:
                    print(f"Lastfm tracks: {track_count}", end="\r")
                    csvfile.flush()
                    time.sleep(0.2)
                    #get_preview_urls()
                if cluster_count >= cluster_size:
                    break

"""def get_preview_urls():
    print("Getting preview urls...")
    df = pd.read_csv(dataset_name)
    df = df[~df['previewURL'].fillna('').str.startswith('http')]
    df = df.dropna(subset=['artist', 'name', 'trackID'])
    print(len(df))

    trackID_to_preview_url = {}
    for index, row in df.iterrows():
        track_id = row.iloc[2]
        preview_url = get_spotify_preview_url(track_id)
        if preview_url:
            trackID_to_preview_url[track_id] = preview_url
        else:
            trackID_to_preview_url[track_id] = 'no_preview'
        if index%100 == 0:
            print(f"Processed {index} tracks", end="\r")
            df["previewURL"] = df["trackID"].map(trackID_to_preview_url)
            df.to_csv(dataset_name, index=False)

    df["previewURL"] = df["trackID"].map(trackID_to_preview_url)
    df.to_csv(dataset_name, index=False)"""



"""def get_preview_urls(n_processes=100, parallel=False):
    print("Getting preview urls...")

    full_df = pd.read_csv(dataset_name)
    #print(full_df.head())
    full_df['previewURL'] = full_df['previewURL'].fillna('None')
    full_df = full_df[~full_df['previewURL'].str.startswith('http') & (full_df['previewURL'] != 'no_preview')]
    full_df = full_df.dropna(subset=['artist', 'name', 'trackID'])
    #print(full_df.head())
    print(len(full_df))

    #split intto n_processes datasets
    df_list = np.array_split(full_df, n_processes)
    for i, df in enumerate(df_list):
        tracks_json = []
        for index, row in df.iterrows():
            tracks_json.append({
                    "name": row.iloc[0],
                    "artist": row.iloc[1],
                    "spotify_track_id": row.iloc[2]
            })
    
        with open(f"json/tracks_{i}.json", 'w', encoding='utf-8') as f:
            json.dump(tracks_json, f, indent=2)
    
    processes = []
    for i in range(n_processes):
        node_cmd = ["node", "preview_finder.js"]
        env = os.environ.copy()
        env["TRACKS_IDX"] = str(i)
        print(f'running preview finder for subprocess {i}')
        if parallel:
            proc = subprocess.Popen(node_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd="node")
            time.sleep(1)
            processes.append(proc)
        else:
            subprocess.run(node_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd="node")
            time.sleep(0.5)
    
    # Wait for all processes to complete
    if parallel:
        for i, proc in enumerate(processes):
            stdout, stderr = proc.communicate()
            #print(f'Subprocess {i} output:')
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
    df.to_csv(dataset_name, index=False)"""
    
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
    if not os.path.exists('.resume'):
        with open(dataset_name, 'w', newline='', encoding='utf-8') as csvfile: #set up csv file with headers
            writer = csv.writer(csvfile)
            writer.writerow(['name', 'artist', 'trackID', 'clusterID', 'clusterType', 'previewURL'])

    #resume from checkpoint
    if os.path.exists('.resume'):
        with open('.resume', 'r') as f:
            content = f.read()
            func_name = content.split(':')[0].strip()
            print(func_name)
            if 'u' not in func_name:
                start_query = int(content.split(':')[1])
        if func_name == 'p':
            get_playlist_tracks(sp, cluster_size=4, start_query=start_query) 
            get_album_tracks(sp, cluster_size=4) 
            get_lastfm_tracks(sp, n_tracks=200000, cluster_size=4) 
            #get_preview_urls()
        elif func_name == 'a':
            get_album_tracks(sp, cluster_size=4, start_query=start_query)
            get_lastfm_tracks(sp, n_tracks=200000, cluster_size=4) 
            #get_preview_urls()
        elif func_name == 'l':
            get_lastfm_tracks(sp, n_tracks=200000, cluster_size=4, start_query=start_query)
            #get_preview_urls()
        #elif func_name == 'u':
            #get_preview_urls()

    else:
        get_playlist_tracks(sp, cluster_size=4) #default 95,000 tracks
        get_album_tracks(sp, cluster_size=4) #default 52,000 tracks
        get_lastfm_tracks(sp, n_tracks=200000, cluster_size=4) #default 304,144 tracks
        #get_preview_urls()

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
