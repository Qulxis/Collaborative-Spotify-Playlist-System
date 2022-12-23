# Spotify-Collaborative-Playlist-Generator
Big data project by: [Andrew](https://github.com/Qulxis), [Alban](https://github.com/Alban999), and [Kenneth](https://github.com/Kennethm-spec). Playlist collaboration project for EECS6893


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About](#about-the-project)
* [Built With](#built-with)
* [Instructions](#instructions)

## About The Project

A Flask web app used to generate a Spotify playlist based on selected tracks and personal preferences.

The purpose of this project is to create a Spotify playlist generator. The users provide reference playlists in their existing libraries. Our model then looks over the distribution of the playlists to train a model that selects the best songs to include in the newly generated playlist.
Our model is novel in that it:
- Does not use the existing "recommend" function from the spotify API. Most existing frameworks simply call that function and do not do any modeling of their own
- Composes a playlist of songs that the users have not already included in their existing playlist or liked songs. This differs our project from Spotify's "Blend" functionality
- Used multiple models to observe the best fit for the specific task

The project files in this repo are to be deployed locally. The interface uses a Flask framework as the UI and the service can be ran as a local host. Note that the flask host (not the users logging in to use the app) will need to include their developer API information from Spotify Developer which can be found at https://developer.spotify.com/. This process is described below in "Add Spotify Authentication".

### Built With
* [GCP] - computing
* [PySpark] - data processing
* [FIRESTORE](https://firebase.google.com/) - datebase
* [Flask](https://flask.palletsprojects.com/en/1.1.x/) - backend
* [Bootstrap](https://getbootstrap.com) - frontend
* [JQuery](https://jquery.com) - frontend
* [Spotify API](https://developer.spotify.com/documentation/web-api/)
* [SKLEARN] - ML modeling snd output
* [CSS] for styling
* [HTML] veiw port rendering
* [Python] script language (Developed for versions 3.7-3.9)

## Instructions:
Clone using:

`git clone https://github.com/Qulxis/spotify-big-data-project.git` 

Create a virtual environment for the project and activate it:

`virtualenv venv`

`source venv/bin/activate`

Install the required packages:

`pip install -r requirements.txt`


### Add Firestore Authentication and Update Paths:
**(Between the dates of 12/22/2022 and 01/02/2022, an existing authentication file will be in place and these steps can be skipped)**

You must have access to a Firestore database collection and its authentication file. For more and setting up a Firestore database on GCP, see [Firebase Authetication](https://firebase.google.com/docs/auth). Once you have the SDK-authentication file, perform the following steps to allow the project to use it:
1. Upload the SDK authentication file to Collaborative-Spotify-Playlist-Generator/app/backend/auth/ . (Note if using the orignal SDK file, steps 2&3 are not required for Firestore setup).
2. Copy the relative path to that SDK file, 'app/backend/auth/YOUR_SDK_FILENAME_HERE.json', and then update the variable "AUTH_PATH" to this path as a string in Collaborative-Spotifpy-Playlist-Generator/.env.
3. Save all changes to files.
4. In your Firestore account, add metadata for the collection of track in a collection which you should label as "bigdata2". We have included an example 100k set of tracks in "100k_demo.csv" which you can upload to your Firestore database using rowy.io [Guide to csv and Rowy.io](https://www.rowy.io/blog/import-csv-to-firestore).

### Add Spotify Authentication:
This project requires Spotify Developer Authentication.
**There are 2 options:** 


*Option 1:*
1. Send us the usernames (emails associated with the accounts) of the Spotify accounts that will be using this app to be added to the whitelist. (TA's for the class bigdata should have received an email regarding this and alternative options to use accounts already included in the whitelist).

OR 

*Option 2:*
1. Setup a [Spotify Developer Account and Project ](https://developer.spotify.com/dashboard) to add accounts, then update the CLIENT_ID and CLIENT_SECRET variables in Spotifpy-Playlist-Generator/.env accordingly.
2. In your Spotify Developer Dashboard for your project, add "http://127.0.0.1:8002/callback" and "http://127.0.0.1:8002/callback/q" to your list of accepted redirects.

### Run the app!
1. Run the file "run.py" in the folder "Collaborative-Spotify-Playlist-Generator/".
2. Access the url "http://127.0.0.1:8002/" on your local machine and follow along on the user interface. Note that generation steps may take longer given large playlists.
3. Enjoy making your playlists! 
## Breakdown
A breakdown of our process is contained in the Jupyter Notebook files in src. We have demos for using the backend, using Spotipy,visualization and EDA, modeling, and Spark processing of the reference data using Spark.
