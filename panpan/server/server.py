from base64 import urlsafe_b64encode
from io import BytesIO
import random
from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Union
from argon2 import hash_password, verify_password
from captcha.image import ImageCaptcha
from PIL import ImageFile, Image

app = FastAPI()
img_captcha = ImageCaptcha()
gen_dict = 'ABCEFGHJKMNPQRTWXYZ23478'


class UserObject(BaseModel):
  uname: str
  passwd: str
  captcha_id: str
  captcha_txt: str


class TokenObject(BaseModel):
  token: str


class CaptchaObject(BaseModel):
  id: str
  img: bytes
  value: Union[str, None]


# @app.post('/token')
def veryfy_token(token: TokenObject):
  return token.token


def rand_captcha_str(length=4):
  return ''.join([random.choice(gen_dict) for _ in range(4)])


@app.get('/captcha/get',
         response_model=CaptchaObject,
         response_model_exclude_unset=True)
def get_captcha(req: Request):
  print(req.client.host)
  captcha_txt = rand_captcha_str()
  img = img_captcha.generate_image(captcha_txt)
  buffer = BytesIO()
  img.save(buffer, format='JPEG')
  buffer = buffer.getvalue()
  buffer = urlsafe_b64encode(buffer).rstrip(b'=')
  return CaptchaObject(id=uuid4().hex, img=buffer)


@app.post('/login')
def login(user: UserObject):
  passwd = hash_password(user.passwd.encode('utf8'))
  return {'log': 'hello'}


@app.post('/register')
def login(token: str = Depends(veryfy_token)):
  return {'token': token}


if __name__ == '__main__':
  import uvicorn
  uvicorn.run('server:app')