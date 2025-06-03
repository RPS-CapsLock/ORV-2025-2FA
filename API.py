from flask import Flask, request, jsonify
import numpy as np
import base64
import io
import os
import main as face_recognition
import subprocess
import uuid
import Mod

app = Flask(__name__)

known_image = face_recognition.load_image_file("known_face.jpg")

def process_info_for_f2a(incoming_image):
    try:
        processed_image = Mod.process_image(incoming_image)
        return processed_image
    except Exception as e:
        print(f"Exception in process_info_for_f2a: {e}")
        return None

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
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json()
    image_b64 = data.get("image")
    user_id = data.get("user_id")

    if not image_b64 or not user_id:
        return jsonify({"error": "Missing image or user_id"}), 400
    try:
        image_data = base64.b64decode(image_b64)
        image = io.BytesIO(image_data)
        from PIL import Image
        incoming_image = Image.open(image).convert("RGB")

        processed_image = process_info_for_f2a(incoming_image)
        if processed_image is None:
            return jsonify({"error": "Image processing failed"}), 500

        result = recognize_face(user_id, processed_image)
        if result is None:
            return jsonify({"error": "Recognition failed"}), 500

        return jsonify({"verified": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500