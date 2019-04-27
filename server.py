from flask import Flask, request, render_template
from flask import send_file,session,jsonify
import string
import random
from infer import *


def id_generator(size=4, chars=string.ascii_lowercase):
	return ''.join(random.choice(chars) for _ in range(size))

app = Flask(__name__,template_folder='static')

@app.route('/static/<path:filename>')
def serve_static(filename):
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(root_dir, 'static'),   filename)

@app.route('/')
def home():
	return send_file('static/index.html')

@app.route('/chat/<bot_id>',methods=['GET', 'POST'])
def chat(bot_id):
	if request.method == 'POST':
		if bot_id in table:
			context = table[bot_id]['context']
			question = request.form.get('ques');
			prev_q = request.form.get('prev_q');
			prev_a = request.form.get('prev_a');
			answer = iq.predict(context,question,prev_q,prev_a)			
			return answer
	if request.method == 'GET':
		if bot_id not in table:
			bot_id = "Oops! Bot not found!"
			bot_im = ""
		else:
			bot_im = table[bot_id]["im_url"]
		return render_template('chat.html',bot=bot_id,bot_im=bot_im)

@app.route('/create_bot',methods=['GET', 'POST'])
def create_bot():

	bot_id = request.form.get('id')
	context = request.form.get('context')
	bot_im = request.form.get('bot_im_url')
	table[bot_id] = {"context":context,"bot_name":bot_id,"im_url":bot_im}

	return SERVER+'/chat/'+bot_id


# SERVER = "10.129.6.41:5000"
table = {}
iq = InferCoQA('model')
print('done loading model ..')

app.run(host='0.0.0.0', debug=True)