from flask import Flask, request, render_template


# Loading Model
from keras.models import load_model
import numpy as np
global model,graph,sess # for predicting the output (creating variables)
import tensorflow as tf

from tensorflow.python.keras.backend import set_session
tf_config = tf.ConfigProto()
sess = tf.Session(config=tf_config)

graph = tf.get_default_graph() # for prediction

set_session(sess)
model = load_model("regressor.h5")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods = ['POST'])
def login():
    a = request.form['a'] # Retriving the name entered in index.html
    b = request.form['b']
    c = request.form['c']
    d = request.form['s']
    # e = str(a)+str(b)+str(c)+str(d)
    if (d == "newyork"):
        s1,s2,s3 = 0,0,1
    if (d == "florida"):
        s1,s2,s3 = 0,1,0
    if (d == "california"):
        s1,s2,s3 = 1,0,0
    
    total = [[s1,s2,s3,a,b,c]]
    with graph.as_default():
        set_session(sess)
        y_pred = model.predict(np.array(total))
        y = y_pred[0][0]
        print(y_pred)
    
    return render_template('index.html', abc = y)

if __name__ == '__main__':
    app.run(debug = True)