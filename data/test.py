from spotify_preview import get_spotify_preview_url

# Pass a Spotify track ID
preview_url = get_spotify_preview_url('1301WleyT98MSxVHPZCA6M')
if preview_url:
    print(f"Preview URL: {preview_url}")
