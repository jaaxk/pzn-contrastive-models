import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
import os

def get_playlist_tracks(sp, n_playlists):
    queries = []
    with open("playlist_queries", "r") as f:
        queries = f.readlines()

    playlists = []
    while len(playlists) < n_playlists:
        query = random.choice(queries)
        results = sp.search(q=query, type="playlist", limit=50, offset=random.randint(0, 1000))
        
        pass

    return tracks

def get_album_tracks():

    return tracks

#Other functions...

def main():

    cluster_size = 5

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")))

    get_playlist_tracks(sp, 10)
    #get_album_tracks()

if __name__ == "__main__":
    main()