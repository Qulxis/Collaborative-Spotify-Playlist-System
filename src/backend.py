import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
default_auth_path = "data/andrew/ecbm4040-ahx2001-firebase-adminsdk-hpfc4-3bffe3cdfc.json"

def add_playlists(playlists,collection, auth=default_auth_path):
    """
    Inputs:
    - playlists:
    Playlist must be a list of dictionaries in the form:
        'playlist_name': playlist['name'],
        'playlist_url': playlist['external_urls']['spotify'],
        'playlist_img_url': playlist['images'][0]['url'],
        'playlist_tracks_url': playlist['tracks']['href'],
        'playlist_id': playlist['id'],
        'playlist_tracks': self._get_playlist_tracks(auth_header, playlist['id'])
    Songs must be in format of:
        'track_artist': track['track']['artists'][0]['name'],
        'track_name': track['track']['name'],
        'track_image': track['track']['album']['images'][0]['url'],
        'track_url': track['track']['external_urls']['spotify'],
        'track_id': track['track']['id']
        }
    collection (str): collection to add playlists to
    auth (str): path to auth file.
    Output:
    - None
    
    """
    try: 
        cred = credentials.Certificate(auth)
        firebase_admin.initialize_app(cred)
    except:
        print("Authentication already loaded")
    db = firestore.client()

    for record in playlists:
        doc_ref = db.collection(collection).document(record['playlist_name'])
        doc_ref.set(record)
    return

def clear_collection(collection, auth=default_auth_path):
    """
    Inputs:
    - collection (str): collection set to clear
    - auth (str): path to auth file.
    Output:
    - None
    
    """
    try: 
        cred = credentials.Certificate(auth)
        firebase_admin.initialize_app(cred)
    except:
        print("Authentication already loaded")
    db = firestore.client()

    ref_coll = db.collection(collection)
    docs = ref_coll.stream()
    for doc in docs:
        db.collection(collection).document(doc.id).delete()
        print("Deleting Playlist:", doc.id )
    return
def get_playlists(collection,auth=default_auth_path):
    """
    Inputs:
    - collection (str): collection to gather data from
    - auth (str): 
    Output:
    - playlists (arr): each element is a playlist dictionary
    """
    try: 
        cred = credentials.Certificate(auth)
        firebase_admin.initialize_app(cred)
    except:
        print("Authentication already loaded")
    db = firestore.client()
    playlists = [] # returned with all playlist dictionaries
    ref_coll = db.collection(collection)
    docs = ref_coll.stream()
    for doc in docs:
        playlists.append(doc.to_dict())
    return playlists

def get_track_ids(playlists):
    """
    Inputs:
    - playlists (arr): output from get_playlists
    Outputs:
    - song_ids (arr): list/array of all song ids from all playlists. No repeating ids
    """
    song_ids = []
    for playlist in playlists:
        # print(playlist)
        for track in playlist['playlist_tracks']:
            song_ids.append(track['track_id'])
    return song_ids





