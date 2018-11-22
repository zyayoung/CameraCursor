from flask import Flask, request
import sys
from pynput.mouse import Button, Controller

app = Flask(__name__)

mouse = Controller()

# identify user's identity
@app.route("/")
def hello():
    if request.args.get('action') == 'click':
        mouse.click(Button.left, 1)
    return "Hello World!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3575)
