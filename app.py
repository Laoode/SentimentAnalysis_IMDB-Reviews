from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load your trained model and vectorizer
model = joblib.load('my_flask_app\sentiment_model.pkl')
vectorizer = joblib.load("my_flask_app\\tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    print(f"Received review: {review}")
    
    # Transform the input using the loaded vectorizer
    transformed_review = vectorizer.transform([review])
    
    # Predict sentiment using the loaded model
    prediction = model.predict(transformed_review)
    
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    print(f"Prediction: {sentiment}")
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
