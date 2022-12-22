# spotify-big-data-project
Big data project: Andrew, Alban, Kenneth. Playlist collaboration project for EECS6893

The purpose of this project is to create a Spotify playlist generator. The users provide reference playlists in their existing libraries. Our model then looks over the distribution of the playlists to train a model that selects the best songs to include in the newly generated playlist.
Our model is novel in that it:
- Does not use the existing "recommend" function from the spotify API. Most to all existing frameworks simply call that function and do not do any modeling of their own
- Composes a playlist of songs that the users have not already included in their existing playlist or liked songs. This differs our project from Spotify's "Blend" functionality
- Used multiple models to observe the best fit for the specific task

The project files in this repo are to be deployed locally. The interface uses a Flask framework as the UI and the service can be ran as a local host. Note that the host (not the user logging in to use the app) will need to include their developer API information from Spotify Developer which can be found at https://developer.spotify.com/. We do not include our own keys as it serves as a security risk.

To start, clone our repo and run ```$ pip install -r requirements.txt```. Then, update the hosting files with your Spotify API keys and connect the desired GCP bucket to store your data. Launch the Flask framework and allow users to login to their accounts to get started on your way to custom playlists! Detailed instructions below:

<!-- TABLE OF CONTENTS -->
<!-- ## Table of Contents

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
* [HTML] veiw port rendering -->

## Instructions:
Clone using:

`git clone ` **** add our git link ****

Create a virtual environment for the project and activate it:

`virtualenv venv`

`source venv/bin/activate`

Install the required packages:

`pip install -r requirements.txt`


### Add Firestore Authentication and Update Paths:
You must have access to a Firestore database collection and its authentication file. For more and setting up a Firestore database on GCP, see (https://firebase.google.com/docs/auth). Once you have the SDK-authentication file, perform the following steps to allow the project to use it:
1. Upload the SDK authentication file to Collaborative-Spotify-Playlist-Generator/app/backend/auth/ . (Note if using the orignal SDK file, steps 2&3 are not required for Firestore setup).
2. Copy the relative path to that SDK file, 'app/backend/auth/YOUR_SDK_FILENAME_HERE.json', and then update the variable "AUTH_PATH" to this path as a string in Collaborative-Spotifpy-Playlist-Generator/.env.
3. Save all changes to files.

### Add Spotify Authentication:
This project requires Spotify Developer Authentication.
There are two options:
1. Send us the usernames (emails associated with the accounts) of the Spotify accounts that will be using this app to be added to the whitelist. (TA's for the class bigdata should have received an email regarding this and alternative options).
OR 
2. Setup a Spotify Developer Account and project (https://developer.spotify.com/dashboard) to add accounts, then update the CLIENT_ID and CLIENT_SECRET variables in Spotifpy-Playlist-Generator/.env accordingly.

### Run the app!
1. Run the file "run.py" in the folder "Collaborative-Spotify-Playlist-Generator/".
2. Access the url "http://127.0.0.1:8002/" on your local machine and follow along on the UI!
3. Enjoy making your playlists! 

