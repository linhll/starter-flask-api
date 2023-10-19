from flask import Flask
import os
import tiktok_captcha
from flask import request, abort

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, world!'

@app.route('/tiktok-captcha/bypass', methods=['POST'])
def bypass_tiktok_captcha():
    if request.method == 'POST':
        body = request.get_json()
        print(body)
        if body == None:
            abort(400)
        images = body.get("images")  
        return {"results": tiktok_captcha.bypass(images)}
    abort(400)
