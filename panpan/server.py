from base64 import urlsafe_b64encode
from http import HTTPStatus
from io import BytesIO
import random
import re
from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException, Request
from argon2 import hash_password, verify_password
from captcha.image import ImageCaptcha
# from PIL import ImageFile, Image
from server_lib import *
from server_lib.defs import SessionObj

app = FastAPI()
img_captcha = ImageCaptcha()
gen_dict = 'ABCEFGHJKMNPQRTWXYZ23478'
session_store = SessionStore()


def rand_captcha_str(length=4):
  return ''.join([random.choice(gen_dict) for _ in range(length)])


@app.get('/captcha/get',
         response_model=CaptchaRes,
         response_model_exclude_unset=True)
def get_captcha(req: BaseReq):
  captcha_txt = rand_captcha_str()
  img = img_captcha.generate_image(captcha_txt)
  buffer = BytesIO()
  img.save(buffer, format='JPEG')
  buffer = buffer.getvalue()
  buffer = urlsafe_b64encode(buffer).rstrip(b'=')
  captcha_id = uuid4()
  session_store.set_session(
      req.uid,
      SessionObj(capt_id=captcha_id, capt_value=captcha_txt),
  )
  return CaptchaRes(id=captcha_id.hex, img=buffer)


@app.get('/captcha/val')
def val_captcha(req: CaptchaReq):
  sobj = session_store.get(req.uid)
  return True


@app.post('/play')
def play(b: CaptchaReq):
  print(b)
  return {"ok": True}


print(__name__)
if __name__ == '__main__':
  import uvicorn
  uvicorn.run('server:app')