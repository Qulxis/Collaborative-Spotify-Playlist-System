from flask import render_template, Blueprint, request, redirect, url_for, session

loading_clearing_dataset_blueprint = Blueprint(
    'loading_clearing_dataset_bp', __name__, template_folder='templates')


@loading_clearing_dataset_blueprint.route("/loading_clearing_dataset", methods=['GET', 'POST'])
def loading():
    if request.method == 'GET':
        return render_template('loading_clearing_dataset.html')
