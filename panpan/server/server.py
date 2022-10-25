from base64 import urlsafe_b64encode
from io import BytesIO
import random
from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException, Request
from argon2 import hash_password, verify_password
from captcha.image import ImageCaptcha
# from PIL import ImageFile, Image
from ..defs import UserObject, CaptchaObject, TokenObject
from .session_store import SessionStore

app = FastAPI()
img_captcha = ImageCaptcha()
gen_dict = 'ABCEFGHJKMNPQRTWXYZ23478'
session_store = SessionStore()


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
  captcha_id = uuid4()
  session_store.set_session(req.client.host,
                            CaptchaObject(captcha_id.hex, buffer, captcha_txt))
  return CaptchaObject(id=captcha_id.hex, img=buffer)


@app.get('/captcha/val')
def val_captcha(req: Request, captcha: CaptchaObject):
  host_ip = req.client.host
  session_captcha = session_store.get(host_ip)
  if session_captcha.id != captcha.id:
    return False
  return session_captcha.value == captcha.value


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