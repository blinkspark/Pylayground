from base64 import urlsafe_b64encode
from datetime import datetime, timedelta
from http import HTTPStatus
from io import BytesIO
import random
import re
from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException, Request
from argon2 import hash_password, verify_password
from captcha.image import ImageCaptcha
# from PIL import ImageFile, Image
from server_lib.defs import *
from google.cloud.firestore import AsyncClient

app = FastAPI()
img_captcha = ImageCaptcha()
gen_dict = 'ABCEFGHJKMNPQRTWXYZ23478'
store = AsyncClient.from_service_account_json('account.json')
session_col = store.collection('session')


def rand_captcha_str(length=4):
  return ''.join([random.choice(gen_dict) for _ in range(length)])


@app.get('/captcha/get',
         response_model=CaptchaRes,
         response_model_exclude_unset=True)
async def get_captcha(req: BaseReq):
  captcha_txt = rand_captcha_str()
  img = img_captcha.generate_image(captcha_txt)
  buffer = BytesIO()
  img.save(buffer, format='JPEG')
  buffer = buffer.getvalue()
  buffer = urlsafe_b64encode(buffer).rstrip(b'=')
  captcha_id = uuid4()
  data = SessionObj(
      capt_id=captcha_id.hex,
      capt_value=captcha_txt,
      expires=datetime.utcnow() + timedelta(hours=1),
  )
  await session_col.document(req.uid).set(dict(data))
  return CaptchaRes(capt_id=captcha_id.hex, capt_img=buffer)


@app.post('/captcha/val')
async def val_captcha(req: CaptchaReq):
  doc = await session_col.document(req.uid).get()
  doc = doc.to_dict()
  return req.capt_id == doc['capt_id'] and req.capt_value == doc['capt_value']


if __name__ == '__main__':
  import uvicorn
  uvicorn.run('server:app')