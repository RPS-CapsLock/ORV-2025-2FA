from flask import Flask, request, jsonify
import numpy as np
import base64
import io
import os

app = Flask(__name__)

known_image = face_recognition.load_image_file("known_face.jpg")


@app.route("/verify", methods=["POST"])
def verify():
