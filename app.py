from flask import Flask, render_template, request
from predict import predict_tags

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    predictions = None
    title = ""
    content = ""

    if request.method == "POST":

        title = request.form["title"]
        content = request.form["content"]

        predictions = predict_tags(title + " " + content, top_k=10)

    return render_template(
        "index.html",
        predictions=predictions,
        title=title,
        content=content
    )

if __name__ == "__main__":
    app.run(debug=True)