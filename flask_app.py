from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        # Sending a POST request to FastAPI's predict route
        try:
            response = requests.post("http://127.0.0.1:8080/predict", json={"text": text})
            summary = response.json() if response.status_code == 200 else "Error"
        except Exception as e:
            summary = f"Error: {str(e)}"
        return render_template('index.html', summary=summary)
    return render_template('index.html', summary=None)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
