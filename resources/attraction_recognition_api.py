import json
import mimetypes
import os

from flask import request
from flask_restful import Resource
# 安裝套件 pip install google-genai
# 匯入套件
from google import genai


ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "AttractionName": {"type": "STRING"},
        "Country": {"type": "STRING"},
        "City": {"type": "STRING"},
        "Town": {"type": "STRING"},
        "Description": {"type": "STRING"},
        "Latitude": {"type": "STRING"},
        "Longitude": {"type": "STRING"},
    },
    "required": [
        "AttractionName",
        "Country",
        "City",
        "Town",
        "Description",
        "Latitude",
        "Longitude",
    ],
}


class AttractionImageRecognition(Resource):
    def post(self):
        image = request.files.get("image")
        if image is None or image.filename == "":
            return {"error": "請選擇景點圖片"}, 400

        mime_type = image.mimetype or mimetypes.guess_type(image.filename)[0]
        if mime_type not in ALLOWED_MIME_TYPES:
            return {"error": "只支援 JPG、PNG、WEBP、HEIC 或 HEIF 圖片"}, 400

        # 讀取 API 金鑰
        # 在 PowerShell 中設定金鑰 $env:GEMINI_API_KEY = "你的金鑰"
        # 金鑰到 Google AI Studio 申請
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "伺服器尚未設定 GEMINI_API_KEY"}, 500

        try:
            # 建立 client 物件
            client = genai.Client(api_key=api_key)
            # 把圖片包裝成 Gemini 看得懂的格式
            image_part = genai.types.Part.from_bytes(
                data=image.read(),
                mime_type=mime_type,
            )
            # 呼叫 Gemini API 進行景點辨識
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                # 組合送給模型的內容：圖片 + 文字提示（prompt）
                contents=[
                    image_part,
                    (
                        "請辨識這張照片中的景點，只回傳符合指定欄位的 JSON。"
                        "無法確認的欄位請填空字串，所有欄位都必須是字串，使用繁體中文回應。"
                    ),
                ],
                # 強制回傳結構化 JSON
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
            # 把回傳的文字字串轉成 Python 字典
            result = json.loads(response.text)
        except json.JSONDecodeError:
            return {"error": "Gemini 回傳的內容不是有效 JSON"}, 502
        except Exception as error:
            return {"error": f"景點辨識失敗：{error}"}, 502

        return result, 200
