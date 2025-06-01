from flask import Flask, request, jsonify
import numpy as np
import base64
import io
import os
import main as face_recognition
import subprocess
import uuid

app = Flask(__name__)

known_image = face_recognition.load_image_file("known_face.jpg")

def process_info_for_f2a():
    pass

def recognize_face(incoming_image):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    incoming_image.save(temp_filename)
    pass


@app.route("/verify", methods=["POST"])
def verify():
    pass