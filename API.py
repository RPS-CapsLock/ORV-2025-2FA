from flask import Flask, request, jsonify
import base64
import io
import os
from face_recognition import use, train
import uuid

app = Flask(__name__)

def recognize_face(user_id, incoming_image):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    incoming_image.save(temp_filename)
    try:
        return use(user_id, temp_filename)
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

        result = train(user_id)

        if result == False:
            return jsonify({"error": "Training failed"}), 500

        return jsonify({"message": f"Training successful for {user_id}."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)