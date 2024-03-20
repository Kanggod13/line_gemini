from flask import Flask, send_file, request , jsonify
import tensorflow as tf
from keras.models import load_model
import os
from PIL import Image
import google.generativeai as genai
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage ,TextSendMessage
from linebot.exceptions import InvalidSignatureError
from werkzeug.utils import secure_filename
import requests


app = Flask(__name__)
UPLOAD_FOLDER = './uploads'

line_bot_api = LineBotApi('Su4j1pJuIXs4HBmMVvQ70IcgV1FVGQjfd0UnJreOaYEwmknf9/n3sDNJdkOfhBF6GHyxLPnC2PbqFYil6I8lW/uAWvl/kd3ZDpkrNXb8mB5YEjvibVGQt3sCO+uPzFxyEM9fVFp0anPgc2BVVpa6QgdB04t89/1O/w1cDnyilFU=','7d8e2a54703799f743618b1a222703ed ')
handler = WebhookHandler('7d8e2a54703799f743618b1a222703ed')
# line_bot_reply = 'https://api.line.me/v2/bot/message/reply'
LINE_API = 'https://api.line.me/v2/bot/message/reply'

model = load_model(
    os.path.join("model\model_20class_20ep_lastest_v1.h5"),
    compile=False,
)

datadict = {
    0: "BooPadPongali",
    1: "FriedChicken",
    2: "GaengKeawWan",
    3: "GoongObWoonSen",
    4: "HoyKraeng",
    5: "HoyLaiPrikPao",
    6: "Joke",
    7: "KaoMooDang",
    8: "KhaoMokGai",
    9: "KkaoKlukKaphi",
    10: "KorMooYang",
    11: "KuaKling",
    12: "LarbMoo",
    13: "MooSatay",
    14: "NamTokMoo",
    15: "PadPakBung",
    16: "PadThai",
    17: "Somtam",
    18: "TomKhaGai",
    19: "TomYumGoong",
}

datadict_th = {
    "BooPadPongali": "ปูผัดผงกระหรี่",
    "FriedChicken": "ไก่ทอด",
    "GaengKeawWan": "แกงเขียวหวาน",
    "GoongObWoonSen": "กุ้งอบวุ้นเส้น",
    "HoyKraeng": "หอยแครง",
    "HoyLaiPrikPao": "หอยลายผัดพริกเผา",
    "Joke": "โจ๊ก",
    "KaoMooDang": "ข้าวหมูแดง",
    "KhaoMokGai": "ข้าวหมกไก่",
    "KkaoKlukKaphi": "ข้าวคลุกกระปิ",
    "KorMooYang": "คอหมูย่าง",
    "KuaKling": "คั่วกลิ้ง",
    "LarbMoo": "ลาบหมู",
    "MooSatay": "หมูสะเต๊ะ",
    "NamTokMoo": "น้ำตกหมู",
    "PadPakBung": "ผัดผักบุ้ง",
    "PadThai": "ผัดไทย",
    "Somtam": "ส้มตำ",
    "TomKhaGai": "ต้มข่าไก่",
    "TomYumGoong": "ต้มยำกุ้ง",
}

def process_image(image):
    try:
        img = Image.open(image)
        if img.mode == "RGBA":
            img = img.convert("RGBA").convert("RGB")
        img = img.resize((224, 224))
        img_array = tf.convert_to_tensor(img, dtype=tf.float32)
        img_array = tf.expand_dims(img_array, 0)

        return img_array
    
    except Exception as e:
        print(f"An error occurred from process_image function: {e}")
        return None
    
@app.route("/uploads/<filename>")
def get_uploaded_image(filename):
  image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  return send_file(image_path, mimetype='image/jpeg')

def predict_image(image_content):
    try:
        # บันทึกรูปภาพที่ได้รับจากไลน์
        image_path = os.path.join(UPLOAD_FOLDER, 'image.jpg')
        with open(image_path, 'wb') as f:
            for chunk in image_content.iter_content():
                f.write(chunk)
        # โหลดรูปภาพและทำนายด้วยโมเดล
        image_data = process_image(image_path)
        prediction = model.predict(image_data, use_multiprocessing=True)
        predicted_class = tf.argmax(prediction, axis=1)[0]
        predicted_class_name = datadict[int(predicted_class)]
        confidence = tf.reduce_max(prediction)
        confidence_percentage = int(confidence * 100)
        chat_response = send_prompt_to_gemini(predicted_class_name)
        return {"predicted_class_name": predicted_class_name, "confidence_percentage": confidence_percentage, "chat_response": chat_response}
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return {"error_msg": "An error occurred during prediction."}

    
@app.route("/", methods=["POST"])
def handle_image_upload():
  if request.method == "POST":
    if 'image' not in request.files:
      return jsonify({"error": "No image uploaded!"})
    
    image = request.files['image']
    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    # เพิ่มเติม: ประมวลผลรูปภาพ (เช่น เปลี่ยนขนาด, แปลงรูปแบบ)
    return jsonify({"message": "Image uploaded successfully!"})

# @app.route('/predict', methods=['POST'])
# def predict():
    
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     try:
#         file = request.files['file']
#         image_data = process_image(file)
#         prediction = model.predict(image_data, use_multiprocessing=True)
#         predicted_class = tf.argmax(prediction, axis=1)[0]
#         predicted_class_name = datadict[int(predicted_class)]
#         confidence = tf.reduce_max(prediction)
#         confidence_percentage = int(confidence * 100)
#         chat_response = send_prompt_to_gemini(predicted_class_name)

#         return {"predicted_class_name": predicted_class_name, "confidence_percentage":confidence_percentage , "chat_response": chat_response}
    
#     except Exception as e:
#         return {"error_msg": e}

@app.route("/webhook", methods=["POST"])
def webhook():
    """รับข้อความและรูปภาพจาก LINE และส่งไปยังโมเดลสำหรับการทำนาย"""
    events = request.get_json()["events"]
    response_list = []  # สร้างรายการเพื่อเก็บข้อความตอบกลับ
    for event in events:
        if event["type"] == "message":
            reply_token = event["replyToken"]
            message_type = event["message"]["type"]
            if message_type == "image":
                message_id = event["message"]["id"]
                # ดึงข้อมูลรูปภาพจากไลน์
                image_content = line_bot_api.get_message_content(message_id)
                # ส่งรูปภาพไปยังโมเดลเพื่อการทำนาย
                prediction = predict_image(image_content)
                # สร้างข้อความตอบกลับ
                reply_text = f"อาหารที่คุณส่งมาคือ {prediction['predicted_class_name']}\n\n{prediction['chat_response']}"
                # เพิ่มข้อความตอบกลับลงในรายการ
                response_list.append({"reply_token": reply_token, "reply_text": reply_text})
            else:
                reply_text = "ไม่รองรับประเภทข้อความนี้"
                # เพิ่มข้อความตอบกลับลงในรายการ
                response_list.append({"reply_token": reply_token, "reply_text": reply_text})
    # ส่งข้อความตอบกลับไปยัง LINE ทีละข้อความ
    for response in response_list:
        line_bot_api.reply_message(response["reply_token"], TextSendMessage(text=response["reply_text"]))
    
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    # ดึง image id
    image_id = event.message.id
    # ดึงรูปภาพจาก LINE
    image_content = line_bot_api.get_message_content(image_id)
    # บันทึกรูปภาพ
    with open('image.jpg', 'wb') as f:
        f.write(image_content)
    # เรียกใช้ฟังก์ชั่น predict_image กับรูปภาพที่บันทึกไว้
    prediction = predict_image('image.jpg')
    # ส่งข้อความตอบกลับผู้ใช้
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"อาหารที่คุณส่งมาคือ {prediction['predicted_class_name']}\n\n{prediction['chat_response']}"))
  
def send_prompt_to_gemini(predicted_class_name):
    # api_key = os.getenv("APIKEYGEMINI")
    api_key = "AIzaSyBUGkOY8UWEzg9zqd9uwV-ZSACuw8P9t14"
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 5000,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    try:
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        prompt_parts = [
            f"Please provide a {predicted_class_name} with the following details: Name of the dish , List of ingredients with amounts , Step-by-step cooking method/instructions , Level of spiciness (mild, medium, hot, etc.), Approximate nutritional information (calories, protein, fat, carbs, etc.) per serving. I'm looking for an authentic and flavorful Thai recipe that covers all those components. If possible, please also mention any dietary restrictions or allergies the recipe can accommodate (vegetarian, gluten-free, nut-free, etc.). I want my response to be in markdown format, with Name of the dish using ### and other topics using ## and emoji to decorate the topics.",
        ]

        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Please try again gemini is reconnecting..."
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
