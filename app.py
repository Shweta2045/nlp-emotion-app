from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# 1. Pipeline components ko load karein
with open("nlp_emotion_pipeline.pkl", 'rb') as file:
    saved_pipeline = pickle.load(file)

tfidf = saved_pipeline['vectorizer']
model_sentiment = saved_pipeline['sentiment_model']
model_emotion = saved_pipeline['emotion_model']
le_sentiment = saved_pipeline['sentiment_encoder']
le_emotion = saved_pipeline['emotion_encoder']

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        user_text = request.form.get('user_text', '').strip()
        
        if user_text:
            # Text ko TF-IDF me badlein
            vectorized_text = tfidf.transform([user_text])
            
            # Sentiment Prediction & Confidence
            sent_class = model_sentiment.predict(vectorized_text)[0]
            sent_probs = model_sentiment.predict_proba(vectorized_text)[0]
            sent_label = le_sentiment.inverse_transform([sent_class])[0]
            sent_conf = np.max(sent_probs) * 100
            
            # Emotion Prediction & Confidence
            emo_class = model_emotion.predict(vectorized_text)[0]
            emo_probs = model_emotion.predict_proba(vectorized_text)[0]
            emo_label = le_emotion.inverse_transform([emo_class])[0]
            emo_conf = np.max(emo_probs) * 100
            
            # Result dictionary backend se UI pe bhejne ke liye
            result = {
                'text': user_text,
                'sentiment': sent_label.upper(),
                'sentiment_conf': f"{sent_conf:.2f}%",
                'emotion': emo_label.upper(),
                'emotion_conf': f"{emo_conf:.2f}%"
            }
            
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)