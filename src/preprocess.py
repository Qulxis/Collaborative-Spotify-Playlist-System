import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

def get_features_from_ids(track_ids, client_id,client_secret, cols_selected = ['acousticness', 'danceability','duration_ms', 'energy', 'instrumentalness', 'key','liveness','loudness','mode','speechiness','tempo','valence']):
    """
    Input: 
    - track_ids (arr): each element is the id for a track on spotify
    - client_id (str): client id from spotify dev
    - client_secret (str): client secret from spotify dev
    Output:
    - df_feature (pandas df): features for each track as selected from cols
    """
    cid = client_id
    secret = client_secret
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager    =    client_credentials_manager)
    track_features = []
    for t_id in track_ids:
        af = sp.audio_features(t_id)
        track_features.append(af)
        tf_df = pd.DataFrame(columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature'])
    for item in track_features:
        for feat in item:
            tf_df = tf_df.append(feat, ignore_index=True)
    tf_df.head()
    
    return tf_df[cols_selected]