import os
import io
from flask import Flask, render_template, request, send_file, flash, redirect, url_for, session
from flask_session import Session
from werkzeug.utils import secure_filename
from tempfile import mkdtemp
from tfc.tfc import TFC

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app = Flask(__name__)
# This may not be secure
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['AUTO_UPDATE'] = True
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = os.urandom(24)

Session(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def main():
    # https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
    if request.method == 'POST':
        # check if the post request has the file part
        if 'sound1' not in request.files or 'sound2' not in request.files:
            flash('Missing file parts')
            return redirect(request.url)
        file1 = request.files['sound1']
        file2 = request.files['sound2']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file1.filename == '' or file2.filename == '':
            flash('Missing selected files')
            return redirect(request.url)
        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            session['file1'] = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            session['file2'] = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            return redirect(url_for('process_audio'))
    return render_template('main.html')


@app.route('/process-audio')
def process_audio():
    flash('processing')
    tfc = TFC()

    output, transition, vis = tfc.process_TFC(session['file1'], session['file2'], 100)

    tfc.write_audio(os.path.join(app.config['UPLOAD_FOLDER'], 'output.wav'), output)
    tfc.write_audio(os.path.join(app.config['UPLOAD_FOLDER'], 'transition.wav'), transition)

    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'vis.svg'), 'wb') as svg:
        svg.write(vis.getvalue())

    return redirect(url_for('return_audio'))


@app.route('/return-audio')
def return_audio():
    flash('return_audio')

    # try:
    #    return send_file(app.config['UPLOAD_FOLDER']+'output.wav', attachment_filename='output.wav')
    # except Exception as e:

    #    return str(e)
    return render_template('result.html',
                           vis=os.path.join(app.config['UPLOAD_FOLDER'], 'vis.svg'),
                           output=os.path.join(app.config['UPLOAD_FOLDER'], 'output.wav'),
                           transition=os.path.join(app.config['UPLOAD_FOLDER'], 'transition.wav'))


if __name__ == '__main__':
    app.run(debug=True)
