from flask import render_template, Blueprint, request, redirect, url_for, session

loading_dataset_blueprint = Blueprint('loading_dataset_bp', __name__, template_folder='templates')


@loading_dataset_blueprint.route("/loading_dataset", methods=['GET', 'POST'])
def loading():
    if request.method == 'GET':
        return render_template('loading_dataset.html')
