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
You must have access to a Firestore database collection and its authentication file. For more, see (https://firebase.google.com/docs/auth). Once you have the SDK-authentication file, perform the following steps to allow the project to use it:
1. Upload the SDK authentication file to Spotify-Playlist-Generator/app/backend/auth/
2. Copy the relative path starting from 'data/' a
