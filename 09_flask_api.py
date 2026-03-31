import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# மாடலை இங்கே லோட் செய்யவும் - இது முக்கியம்!
# கோப்பு பெயர் சரியாக இருப்பதை உறுதி செய்யவும்
try:
    model = joblib.load('./07_output_salepricemodel.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # மாடல் லோட் ஆகவில்லை என்றால் எர்ரர் காட்ட
    if model is None:
        return render_template('index.html', prediction_text="Error: Model file (.pkl) not found or not loaded!")

    try:
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        prediction = model.predict(final_features)
        
        # prediction ஒரு array ஆக இருந்தால் முதல் மதிப்பை எடுக்கவும்
        output = prediction[0] if isinstance(prediction, np.ndarray) else prediction
        formatted_price = "{:,.2f}".format(float(output))

        return render_template('index.html', prediction_text=f'${formatted_price}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(port=5500, debug=True)
