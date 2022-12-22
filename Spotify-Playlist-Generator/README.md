<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About](#about-the-project)
* [Built With](#built-with)

## About The Project

A Flask web app used to generate a Spotify playlist based on selected tracks and personal preferences.

### Built With
* [GCP] - computing
* [PySpark] - data processing
* [FIRESTORE] - datebase
* [Flask](https://flask.palletsprojects.com/en/1.1.x/) - backend
* [Bootstrap](https://getbootstrap.com) - frontend
* [JQuery](https://jquery.com) - frontend
* [Spotify API](https://developer.spotify.com/documentation/web-api/)
* [SKLEARN] - ML modeling snd output
* [CSS] for styling
* [HTML] veiw port rendering


<!-- GETTING STARTED -->
## Getting Started
Clone using:

`git clone ` **** add our git link ****

Create a virtual environment for the project and activate it:

`virtualenv venv`

`source venv/bin/activate`

Install the required packages:

`pip install -r requirements.txt`

## Add Firestore Authentication and Update Paths:
You must have access to a Firestore database collection and its authentication file. For more and setting up a Firestore database on GCP, see (https://firebase.google.com/docs/auth). Once you have the SDK-authentication file, perform the following steps to allow the project to use it:
1. Upload the SDK authentication file to Spotify-Playlist-Generator/app/backend/auth/
2. Copy the relative path to that SDK file, 'app/backend/auth/YOUR_SDK_FILENAME_HERE.json', and then update the variable "AUTH_PATH" to this path as a string in Spotifpy-Playlist-Generator/.env. Do the same on line 7 in Spotify-Playlist-Generator/app/backend/backend.py.
3. Save all changes to files.

## Add Spotify Authentication:
This project requires Spotify Developer Authentication.
There are two options:
1. Send us the usernames of the Spotify accounts that will be using this app (request for this information was sent to TAs).
OR 
2. Setup a Spotify Developer Account and project (https://developer.spotify.com/dashboard), then update the CLIENT_ID and CLIENT_SECRET variables in Spotifpy-Playlist-Generator/.env accordingly.

## Run the app!
1. Run the file "run.py" in the folder "Spotify-Playlist-Generator".
2. Access the url "http://127.0.0.1:8002/" on your local machine and follow along on the UI!
3. Enjoy making your playlists! 