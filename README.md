# spotify-big-data-project
Big data project: Andrew, Alban, Kenneth. Playlist collaboration project for EECS6893

The purpose of this project is to create a Spotify playlist generator. The users provide reference playlists in their existing libraries. Our model then looks over the distribution of the playlists to train a model that selects the best songs to include in the newly generated playlist.
Our model is novel in that it:
- Does not use the existing "recommend" function from the spotify API. Most to all existing frameworks simply call that function and do not do any modeling of their own
- Composes a playlist of songs that the users have not already included in their existing playlist or liked songs. This differs our project from Spotify's "Blend" functionality
- Used multiple models to observe the best fit for the specific task

The project files in this repo are to be deployed locally. The interface uses a Flask framework as the UI and the service can be ran as a local host. Note that the host (not the user logging in to use the app) will need to include their developer API information from Spotify Developer which can be found at https://developer.spotify.com/. We do not include our own keys as it serves as a security risk.

To start, run ```$ pip install -r requirements.txt```. Then, update the hosting files with your Spotify API keys and connect the desired GCP bucket to store your data. Launch the Flask framework and allow users to login to their accounts to get started on your way to custom playlists!
```
├── README.md
├── requirements.txt
├── src
│   ├── APIs
│   │   └── spotipy_access.py
│   ├── DataVisualization.ipynb
│   ├── Organization.md
│   ├── andrew_dev.ipynb
│   ├── data
│   │   ├── README.txt
│   │   └── andrew
│   │       └── andrew_last_2000_songs
│   └── kenneth_dev.ipynb
└── tree.ipynb

4 directories, 10 files
```

