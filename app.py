from flask import Flask, request, jsonify, render_template
from textSummarizer.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

# Root route with a styled HTML form to input text
@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    error = None

    if request.method == "POST":
        text = request.form["text"]
        if not text:
            error = "No text provided for summarization."
        else:
            try:
                # Use the PredictionPipeline to summarize the text
                obj = PredictionPipeline()
                summary = obj.predict(text)
            except Exception as e:
                error = str(e)

    # Render the index.html with summary and error variables
    return render_template("index.html", summary=summary, error=error)

# Predict route for API POST requests
@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        # Expecting a JSON payload with a "text" field
        data = request.get_json()
        text = data.get("text", "")

        # Make sure the input is not empty
        if not text:
            return jsonify({"error": "No text provided for summarization."}), 400

        # Use the PredictionPipeline to summarize the text
        obj = PredictionPipeline()
        summary = obj.predict(text)

        # Return the summary as a JSON response
        return jsonify({"summary": summary}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
