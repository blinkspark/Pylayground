import base64
from io import BytesIO
import random
import string
from fastapi import FastAPI, Depends, Form, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from argon2 import hash_password, verify_password
from captcha.image import ImageCaptcha
from PIL import ImageFile, Image

app = FastAPI()
img_captcha = ImageCaptcha()
gen_dict = 'abcdefgjkmpqrstwxyzABCDEFGJKMPQRSTWXYZ234578'


class UserObject(BaseModel):
  uname: str
  passwd: str
  captcha_id: str
  captcha_txt: str


class TokenObject(BaseModel):
  token: str


# @app.post('/token')
def veryfy_token(token: TokenObject):
  return token.token


def rand_captcha_str(length=4):
  return ''.join([random.choice(gen_dict) for _ in range(4)])


@app.get('/captcha/get')
def get_captcha():
  captcha_txt = rand_captcha_str()
  img = img_captcha.generate_image(captcha_txt)
  buffer = BytesIO()
  buffer.write(b'aa')
  print(buffer)
  img.save(buffer, format='JPEG')
  return {"captcha_img": buffer.read()}


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