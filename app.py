import numpy as np
from flask import Flask, render_template, request
import pickle
import logging

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

logging.basicConfig(filename="log_ver1.log",format='%(asctime)s %(message)s',filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logger.info("Code Executed Successfully")
logger.warning("Its a Warning")



@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]

    # re-arranging the list as per data set
    feature_list = [features[4]] + features[:4] + features[5:11][::-1] + features[11:17][::-1] + features[17:][::-1]
    features_arr = [np.array(feature_list)]

    prediction = model.predict(features_arr)

    print(features_arr)
    print("prediction value: ", prediction)

    result = ""
    if prediction == 1:
        result = "The credit card holder will be Defaulter in the next month"
    else:
        result = "The Credit card holder will not be Defaulter in the next month"

    return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)
