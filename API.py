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

def recognize_face(user_id, incoming_image):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    incoming_image.save(temp_filename)
    try:
        result = subprocess.run(
            ['python', 'script.py', 'use', user_id, temp_filename],
            capture_output=True, text=True
        )
        output = result.stdout.strip()
        print(f"Script output:\n{output}")

        if "match" in output:
            return True
        elif "mismatch" in output:
            return False
        else:
            return None


@app.route("/verify", methods=["POST"])
def verify():
    pass