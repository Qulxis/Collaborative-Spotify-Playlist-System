import json
import requests
from flask import render_template, Blueprint, request, redirect, url_for, session
from app.spotify_api.spotify_handler import SpotifyHandler

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier

from app.backend.backend import *


result_blueprint = Blueprint(
    'result_bp', __name__, template_folder='templates')
spotify_handler = SpotifyHandler()

def extract_letters(string):
    return ''.join([letter for letter in string if not letter.isdigit()])

def get_current_playlists():
    """
    Returns all the current playlist in session storage
    Inputs:
    - None
    Outputs:
    - playlist_data (arr): each element is a playlist    
    """
    authorization_header = session['authorization_header']
    # -------- Get user's name, id, and set session --------
    profile_data = spotify_handler.get_user_profile_data(
        authorization_header)
    user_display_name, user_id = profile_data['display_name'], profile_data['id']
    session['user_id'], session['user_display_name'] = user_id, user_display_name

    try:
        print(session["playlist_data"]) 
        playlist_data = session["playlist_data"]
    except:
        playlist_data = spotify_handler.get_user_playlist_data(
            session['authorization_header'], session['user_id'])
    else:
        playlist_data = session["playlist_data"]
    
    return playlist_data

## Edit this to do all playlists
def get_playlist_tracks(playlists):
    """
    Inputs: 
    - playlist (arr): elements are dictionary format of playlist
    Ouputs:
    - track_ids (arr): each element (str) is a track id code from Spotipy
    - track_names (arr): each element (str) is the name of a track
    """
    track_ids = []
    track_names = []
    for playlist in playlists[:2]: # change to playlist in playlists later for fullset
            for track in playlist['playlist_tracks'][:5]:
                try:
                    track_ids.append(track['track_id'])
                    track_names.append(track['track_name'].encode(encoding='UTF-8',errors='strict'))
                except:
                    print("Can't encode output")

    return track_ids, track_names


def get_features(track_ids):
    """
    This function gets the metadata/features for tracks given a list of their ids
    
    Inputs:
    - track_ids (arr): each element (str) is a track id from Spotify
    Ouputs:
    - features (arr): each element is an arr


    """
    features = []

    for track_id in track_ids:

        try:
            audio_api_endpoint = f"https://api.spotify.com/v1/audio-features/{track_id}"
            audio_features = json.loads(requests.get(
                audio_api_endpoint, headers=session['authorization_header']).text)

            features.append(audio_features)
        except:
            print("Error track")
    return features

def get_info(track_ids): #Formats
    """
    Gives the track info as expected in result.html
    Input 
    - track_ids (list): elements are track ids
    Ouput:
    - features (arr): element is a track dictionary from spotify api: https://developer.spotify.com/console/get-track/
    """
    features = []
    for track_id in track_ids:

        try:
            track_endpoint = f"	https://api.spotify.com/v1/tracks/{track_id}"
            track_features = json.loads(requests.get(
                track_endpoint, headers=session['authorization_header']).text)

            features.append(track_features)
        except:
            print("Error track")
    return features


@result_blueprint.route("/route_to_dataset_clearing", methods=['GET', 'POST'])
def dataset_clearing_callback():
    return redirect(url_for("loading_clearing_dataset_bp.loading"))

@result_blueprint.route("/clear_dataset", methods=['GET', 'POST'])
def clear_dataset():
    """
    This function clears the collection of 'playlists_of_interest' and 'playlists_of_no_interest.
    Returns: redirect
    """
    if request.method == 'POST':
        auth = os.environ['AUTH_PATH']
        try:  # Clearing just for demo. Should only clear to reset all stored data
            clear_collection('playlists_of_interest', auth=auth)
            clear_collection('playlists_of_no_interest', auth=auth)
            print("dataset cleared")
        except:
            print("collections don't exist")

        playlist_data = get_current_playlists()
        user_name = session['user_display_name']

        return render_template('select_tracks.html',
                            user_display_name=user_name,
                            playlists_data=playlist_data,
                            func=extract_letters)
        ###############################################################################

    return redirect(url_for('not_found'))

@result_blueprint.route("/route_to_adding_to_dataset_loading", methods=['GET', 'POST'])
def dataset_addition_callback():
    """
    when adding datasets is intitiated first start up the loading page
    """
    selected_playlists = request.form.get('selected_playlists').split(',')  # Returns playlist ids (array)
    session['selected_playlists'] = selected_playlists

    return redirect(url_for("loading_adding_to_dataset_bp.loading"))

@result_blueprint.route("/add_selected_to_dataset", methods=['GET', 'POST'])
def add_to_dataset():
    """
    This function look at playlist from session's 'selected_playlists' and then uses the exisiting total
    set to then create a set of interesting playlists and not interesting playlists.
    The outputs are stored in the firestore database for further use.
    Last user always has priority to change playlists from interesting to not interesting.

    Inputs:
    - None
    Ouput:
    - Redirect
    """
    selected_playlists = session['selected_playlists']
    print(selected_playlists)

    if request.method == 'POST':
        playlist_data = get_current_playlists()
        playlists_of_interest_name = session['selected_playlists']

        playlists_of_interest = []
        playlists_of_no_interest = []
        # Playlists are in the format described in spotify_handler
        for playlist in playlist_data:
            # If in playlist in list of selected, add to selected (interest), otherwise add to not interest
            if playlist['playlist_id'] in playlists_of_interest_name:
                playlists_of_interest.append(playlist)
            else:
                playlists_of_no_interest.append(playlist)

        ##########################################################################
        #FIRESTORE Processing:
        auth = os.environ['AUTH_PATH']
        try:  # If playlist collections exist, get the data
            final_playlists_of_interest = get_playlists(
                'playlists_of_interest')
            final_playlists_of_no_interest = get_playlists(
                'playlists_of_no_interest')
        except:  # Otherwise, set them to empty lists
            final_playlists_of_interest = []
            final_playlists_of_no_interest = []
        # Give current user priority:
        # If track is in old_interest, remove from interest and add to no_interest and vice versa
        for playlist in playlists_of_interest:
            if playlist in final_playlists_of_no_interest:  # playlists is in old no_interest, we must remove it
                final_playlists_of_no_interest.remove(
                    playlist)  # Remove from old set no_interest
            if playlist not in final_playlists_of_interest:  # Check not in interest set
                # Add to new set interest, overwriting previous
                final_playlists_of_interest.append(playlist)
        for playlist in playlists_of_no_interest:
            if playlist in final_playlists_of_interest:  # playlists is in old interest.
                final_playlists_of_interest.remove(
                    playlist)  # Remove from old set interest
            if playlist not in final_playlists_of_interest:
                final_playlists_of_no_interest.append(playlist)
        # Clear old database selections:
        clear_collection('playlists_of_interest', auth=auth)
        clear_collection('playlists_of_no_interest', auth=auth)

        #Write new Sets:
        playlists_of_interest = final_playlists_of_interest
        playlists_of_no_interest = final_playlists_of_no_interest

        add_playlists(playlists_of_interest,
                      'playlists_of_interest', auth=auth)
        add_playlists(playlists_of_no_interest,
                      'playlists_of_no_interest', auth=auth)
        return redirect(url_for("loading_adding_to_dataset_bp.loading"))
        ###############################################################################

    return redirect(url_for('not_found'))


@result_blueprint.route("/route_to_playlist_generation", methods=['GET', 'POST'])
def your_playlist():
    """
    Main function: 
    - Fetches all existing data from user inputs from firestore and gathers metadata features for each track
    - Generate RF model
    - Gather's reference data (data to select new songs from)
    - Thresholding song limits
    - Creates final playlist in URIs stored in session and in a csv file, tracks_uri.csv


    
    """
    # global tracks_uri
    authorization_header = session['authorization_header']
    auth = os.environ['AUTH_PATH']
    if request.method == 'POST':
        #Read data and access metadata features for each track. Also labels the data for training
        playlists_of_interest = get_playlists('playlists_of_interest',auth=auth)
        playlists_of_no_interest = get_playlists('playlists_of_no_interest',auth=auth)

        print("\n +ve playlists: ", len(playlists_of_interest))
        good_track_ids, good_track_names = get_playlist_tracks(playlists_of_interest)
        print("good tracks",good_track_ids)
            
        bad_track_ids = []
        bad_track_names = []
        print("\n -ve playlists: ", len(playlists_of_no_interest))
        tmp_ids, tmp_names = get_playlist_tracks(playlists_of_no_interest)
        print("bad tracks", tmp_ids)

        for tmp_id, tmp_name in zip(tmp_ids, tmp_names):
            if tmp_id not in good_track_ids and tmp_id not in bad_track_ids:
                bad_track_ids.append(tmp_id)
                bad_track_names.append(tmp_name)

        ratings = [1] * len(good_track_ids) + [0] * len(bad_track_ids)
        track_ids = good_track_ids + bad_track_ids
        track_names = good_track_names + bad_track_names
        print("track ids", track_ids)

        print("\nCalculating ...")
        features = get_features(track_ids)
        favorites_df = pd.DataFrame(features, index=track_names)
        favorites_df['rating'] = ratings
        favorites_df.to_csv('track_features.csv')
        print("column names", favorites_df.columns)
        ############################################################################################
        # MODEL Training:
        training_df = favorites_df[["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence", "rating"]] #REMOVE MODE AND KEY

        print(training_df)

        X_train = training_df.drop('rating', axis=1)
        y_train = training_df['rating']

        X_scaled = StandardScaler().fit_transform(X_train)
        pca = decomposition.PCA().fit(X_scaled)

        variance_ratio = pca.explained_variance_ratio_
        cum_var = np.cumsum(variance_ratio)
        threshold = 0.95
        n_components = next(i for i, v in enumerate(
            cum_var) if v > threshold) + 1

        pca = decomposition.PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_scaled)

        v = TfidfVectorizer(sublinear_tf=True,
                            ngram_range=(1, 6), max_features=10000)
        X_names_sparse = v.fit_transform(track_names)

        X_train = sparse.csr_matrix(
            sparse.hstack([X_train_pca, X_names_sparse]))

        n_splits = 5

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        rfc_parameters = {
            'max_features': [4, 6, 8, 10],
            'min_samples_leaf': [1, 3, 5, 7],
            'max_depth': [3, 5, 7]
        }
        rfc = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, oob_score=True)
        forest_grid = GridSearchCV(
            rfc, rfc_parameters, n_jobs=-1, cv=skf, verbose=1)
        forest_grid.fit(X_train, y_train)
        print("Best score: ", forest_grid.best_score_)

        #############################################################
        # # Using Recommendation system
        #############################################################
        # rec_tracks_per_track = 2
        # max_rec_tracks = 2000
        # rec_tracks_per_track = min([max_rec_tracks, len(
        #     favorites_df['id']) * rec_tracks_per_track]) // len(favorites_df['id'])
        # print(f"Using {rec_tracks_per_track} test tracks per track")

        # print("1111")
        # try:
        #     print("AAAA")
        #     get_reccomended_url = f"https://api.spotify.com/v1/recommendations?limit={100}"
        #     response = requests.get(get_reccomended_url,
        #                             headers=authorization_header,
        #                             params={'seed_tracks': good_track_ids}).text
        #     rec_tracks = json.loads(response)['tracks']
        # except:
        #     print("error recommandation")

        # rec_track_ids = []
        # rec_track_names = []
        # for i in rec_tracks:
        #     rec_track_ids.append(i['id'])
        #     rec_track_names.append(i['name'])

        # rec_features = get_features(rec_track_ids)

        # rec_playlist_df = pd.DataFrame(rec_features, index=rec_track_names)
        # rec_playlist_df.drop_duplicates(subset='id', inplace=True)
        # rec_track_names = rec_playlist_df.index.tolist()
        ####################################################################
        # Load in 100k dataset
        ####################################################################
        try:
            dummy_data = load_reference_data(collection='bigdata2', num_tracks=2000) # Set to top 2000. See docs in backend for more options
            print("Reference data loaded?", not not dummy_data)
            rec_playlist_df = pd.DataFrame(dummy_data, index=False)
            print("columns of rec:", rec_playlist_df.columns)
            rec_playlist_df = rec_playlist_df.rename(
                columns={'durationMs': 'duration_ms'})
            print(rec_playlist_df.head())
            print(rec_playlist_df.columns)
        except:
            print("failed to load reference data first time")
        else:

            dummy_data = load_reference_data(
                collection='bigdata2', num_tracks=1000)
            rec_playlist_df = pd.DataFrame(dummy_data, index=False)
            rec_playlist_df = rec_playlist_df.rename(
                columns={'durationMs': 'duration_ms'})
            print(rec_playlist_df.head())
            print(rec_playlist_df.columns)
        dummy_data = load_reference_data(
            collection='bigdata2', num_tracks=1000)
        rec_playlist_df = pd.DataFrame(dummy_data)
        rec_playlist_df = rec_playlist_df.rename(
            columns={'durationMs': 'duration_ms'})
        print(rec_playlist_df.head())
        print(rec_playlist_df.columns)
        rec_playlist_df.drop_duplicates(subset='id', inplace=True) # remove duplicates
        ###############################################################################################
        # New Track selection
        testing_df = rec_playlist_df[
            [
                "acousticness", "danceability", "duration_ms", "energy",
                "instrumentalness", "liveness", "loudness",
                "speechiness", "tempo", "valence"
            ] #Removed key and mode
        ]
        # rec_track_names = rec_playlist_df.index.tolist()
        print("testing_df",testing_df)
        print(testing_df.columns)
        testing_df_scaled = StandardScaler().fit_transform(testing_df)
        testing_df.to_csv("testing_df.csv") 

        X_test = pca.transform(testing_df_scaled)
        rec_track_names = rec_playlist_df["title"].values.tolist() # Get track names as a list
        
        X_test_names = v.transform(rec_track_names)

        X_test = sparse.csr_matrix(sparse.hstack([X_test, X_test_names]))
        y_pred_final = np.array([1] * X_test_names.shape[0])

        forest_grid.best_estimator_.fit(X_train, y_train)
        y_pred = forest_grid.best_estimator_.predict(X_test)

        y_pred_final = y_pred_final * y_pred
        print("Number of disliked tracks by model: ", sum(y_pred == 0))
        print("Number of disliked tracks: ", sum(y_pred_final == 0))
        print("Number of liked tracks: ", sum(y_pred_final == 1))

        print(y_pred_final)

        final_tracks = rec_playlist_df[y_pred_final.astype(bool)]
        print(final_tracks.columns)
        final_tracks_list = final_tracks.values.tolist() # original into to render_template
     
        print("END")
        

        try:
            print("rec keys:", final_tracks.keys)
        except:
            print("rec keys failed to display, this is a encoding error")
        
        
        
        tracks_uri = [track for track in final_tracks['uri']]
        #Thresholding to top 20 songs
        if len(tracks_uri) > 20:
            tracks_uri = tracks_uri[0:20]
            print("Too many songs selected by model, changing to top 20 uri")
        
        session['tracks_uri'] = tracks_uri
        # Save to df:
        tracks_uri_df = pd.DataFrame(columns=["tracks_uri"], data=tracks_uri)
        tracks_uri_df.to_csv('tracks_uri.csv')
        print("Saved")
        print("tracks_uri",session['tracks_uri'])

        track_ids = final_tracks['id'].values.tolist()
        if len(track_ids) > 20:
            track_ids = track_ids[0:20]
            print("Too many songs selected by model, changign to top 20")
            
        data = get_info(track_ids = track_ids) # Get track formated data        

        session['new_playlist'] = data
        return render_template('result.html', data=data) #changed from results.html -Andrew + Kenneth



    return redirect(url_for('not_found'))



@result_blueprint.route("/save-playlist", methods=['GET', 'POST'])
def save_playlist():
    """
    This function takes the uris of the model-selected track and uses them to generate 
    a playlist in the user's spotify libarary.
    Inputs:
    - None
    Ouputs:
    - Rendered template for final playlist view   
    """
    authorization_header = session['authorization_header']
    user_id = session['user_id']

    playlist_name = request.form.get('playlist_name')
    playlist_data = json.dumps({
        "name": playlist_name,
        "description": "Recommended songs",
        "public": True
    })

    create_playlist_url = f"https://api.spotify.com/v1/users/{user_id}/playlists"

    response = requests.post(create_playlist_url,
                             headers=authorization_header,
                             data=playlist_data).text

    playlist_id = json.loads(response)['id']
    print("session_keys",session)
    tracks_uri_df = pd.read_csv('tracks_uri.csv')
    tracks_uri = tracks_uri_df["tracks_uri"].values.tolist()
    tracks_data = json.dumps({
        "uris": tracks_uri,
    })

    add_items_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    response = requests.post(
        add_items_url, headers=authorization_header, data=tracks_data).text

    return render_template('listen.html', playlist_id=playlist_id)
