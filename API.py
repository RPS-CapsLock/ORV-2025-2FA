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

def recognize_face(user_id, incoming_image):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    incoming_image.save(temp_filename)
    try:
        result = subprocess.run(
            ['python', 'main.py', 'use', user_id, temp_filename],
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

        if incoming_image is None:
            return jsonify({"error": "Image processing failed"}), 500

        result = recognize_face(user_id, incoming_image)
        if result is None:
            return jsonify({"error": "Recognition failed"}), 500

        return jsonify({"verified": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    images_b64 = data.get("images")
    user_id = data.get("user_id")

    if not images_b64 or not user_id:
        return jsonify({"error": "Missing images or user_id"}), 400

    user_folder = os.path.join(".", user_id)
    os.makedirs(user_folder, exist_ok=True)

    try:
        for i, image_b64 in enumerate(images_b64):
            image_data = base64.b64decode(image_b64)
            image = io.BytesIO(image_data)
            from PIL import Image
            img = Image.open(image).convert("RGB")
            img.save(os.path.join(user_folder, f"user_{i}.jpg"))

        result = subprocess.run(
            ['python', 'main.py', 'train', user_id],
            capture_output=True, text=True
        )
        print(result.stdout)
        print(result.stderr)

        if result.returncode != 0:
            return jsonify({"error": "Training failed", "details": result.stderr}), 500

        return jsonify({"message": f"Training successful for {user_id}."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500