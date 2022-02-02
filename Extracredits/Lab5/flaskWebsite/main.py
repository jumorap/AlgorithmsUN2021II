from flask import Flask, render_template, request, redirect, url_for
import chatter


# Initialize the app
app = Flask(__name__)


# Define '/' route
@app.route('/')
def index():
    return render_template('index.html')


# Define '/get' route
@app.route("/get")
def get_bot_response():
    # Get user's input and return the bot's response
    user_message = request.args.get('msg')
    return str(chatter.say_something(user_message))


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
