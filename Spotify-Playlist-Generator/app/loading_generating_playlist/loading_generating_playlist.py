from flask import render_template, Blueprint, request, redirect, url_for, session

loading_generating_playlist_blueprint = Blueprint(
    'loading_generating_playlist_bp', __name__, template_folder='templates')


@loading_generating_playlist_blueprint.route("/loading_generating_playlist", methods=['GET', 'POST'])
def loading():
    if request.method == 'GET':
        return render_template('loading_generating_playlist.html')
