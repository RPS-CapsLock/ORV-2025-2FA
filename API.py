import shutil

from flask import Flask, request, jsonify
import base64
import io
import os
from face_recognition import use, train
import uuid

app = Flask(__name__)

def recognize_face(user_id, incoming_image):
    temp_uuid = uuid.uuid4().hex
    os.makedirs(f'./temp_{temp_uuid}', exist_ok=True)
    temp_filename = f"./temp_{temp_uuid}/temp_{temp_uuid}.jpg"
    incoming_image.save(temp_filename)
    temp_dir = f"./temp_{temp_uuid}"
    try:
        return use(user_id, temp_uuid)
    finally:
        if os.path.exists(temp_filename):
             shutil.rmtree(temp_dir)


@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json()

    image_b64 = data.get("image")
    user_id = data.get("user_id")

    if not image_b64 or not user_id:
        return jsonify({"error": "Missing image or user_id"}), 400

    if image_b64.startswith("data:image/"):
        image_b64 = image_b64.split(",", 1)[1]

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
def train_route():
    data = request.get_json()
    images_b64 = data.get("images")
    user_id = data.get("user_id")

    if not images_b64 or not user_id:
        return jsonify({"error": "Missing images or user_id"}), 400

    user_folder = os.path.join(".", str(user_id))
    os.makedirs(user_folder, exist_ok=True)

    try:
        from PIL import Image

        for i, image_b64 in enumerate(images_b64):
            if image_b64.startswith("data:image/"):
                image_b64 = image_b64.split(",", 1)[1]

            image_data = base64.b64decode(image_b64)
            image = io.BytesIO(image_data)
            img = Image.open(image).convert("RGB")
            img.save(os.path.join(user_folder, f"user_{i}.jpg"))

        result = train(user_id)

        if not result:
            return jsonify({"error": "Training failed"}), 500

        return jsonify({"message": f"Training successful for the {user_id}."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)