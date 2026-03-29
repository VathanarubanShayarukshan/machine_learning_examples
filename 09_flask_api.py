import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# மாடலை லோட் செய்யவும்
try:
    model = joblib.load('./07_output_salepricemodel.pkl')
except:
    print("Error: .pkl file not found! Run 07_model.py first.")

@app.route('/')
def home():
    # ஆரம்பத்தில் எந்த முடிவும் காட்டாமல் வெற்றுப் பக்கத்தைக் காட்டும்
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form-ல் இருந்து 11 தகவல்களையும் எடுக்கிறது
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # கணிப்பு (Prediction)
        prediction = model.predict(final_features)
        
        # விலையை கமாக்களுடன் மாற்ற (எ.கா: $170,824,756.08)
        formatted_price = "{:,.2f}".format(prediction[0])

        return render_template('index.html', prediction_text=f'${formatted_price}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(port=5500, debug=True)
