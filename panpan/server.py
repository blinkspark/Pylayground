from base64 import urlsafe_b64encode
from datetime import datetime, timedelta
from http import HTTPStatus
from io import BytesIO
import random, os
from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException, Request, Header
from argon2 import hash_password, verify_password
from captcha.image import ImageCaptcha
# from PIL import ImageFile, Image
from server_lib.defs import *
from google.cloud.firestore import AsyncClient
from jose import jwt

app = FastAPI()
img_captcha = ImageCaptcha()
gen_dict = 'ABCEFGHJKMNPQRTWXYZ23478'
store = AsyncClient.from_service_account_json('account.json')
session_col = store.collection('session')
user_col = store.collection('user')
BAD_REQ = HTTPStatus.BAD_REQUEST
NEED_X_UID = "Need X_UID"
CAPTCHA_ERR = "Captcha validation failed"
USER_EXISTS = "User already exist"
USER_OR_PASS_ERR = "User or password is wrong"
TOKEN_ERR = "Access token error"
data_folder = os.environ.get('X_DATA_FOLDER', 'data')
if not os.path.exists(data_folder):
  os.makedirs(data_folder)


def rand_captcha_str(length=4):
  return ''.join([random.choice(gen_dict) for _ in range(length)])


@app.get('/captcha',
         response_model=CaptchaRes,
         response_model_exclude_unset=True)
async def get_captcha(x_uid: Union[str, None] = Header(default=None)):
  if x_uid == None:
    raise HTTPException(BAD_REQ, NEED_X_UID)
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
      expires=datetime.utcnow(),
  )
  await session_col.document(x_uid).set(dict(data))
  return CaptchaRes(capt_id=captcha_id.hex, capt_img=buffer)


# @app.post('/captcha/val')
async def val_captcha(req: CaptchaReq,
                      x_uid: Union[str, None] = Header(default=None)):
  if x_uid == None:
    raise HTTPException(BAD_REQ, NEED_X_UID)
  doc = await session_col.document(x_uid).get()
  doc = doc.to_dict()
  return req.capt_id == doc['capt_id'] and req.capt_value == doc['capt_value']


@app.post('/user/register',
          response_model=UserRes,
          response_model_exclude_unset=True)
async def register(req: UserReq, ok: bool = Depends(val_captcha)):
  if not ok:
    raise HTTPException(BAD_REQ, CAPTCHA_ERR)
  passwd = hash_password(req.passwd.encode('utf8'))
  key = os.urandom(32)
  token = jwt.encode({'exp': datetime.utcnow() + timedelta(days=14)}, key=key)
  docs = await user_col.where('username', '==', req.uname).limit(1).get()
  if len(docs) > 0:
    raise HTTPException(BAD_REQ, USER_EXISTS)
  await user_col.add({
      'username': req.uname,
      'password': passwd,
      'key': key,
      'token': token
  })
  return UserRes(uname=req.uname, token=token)


@app.post('/user/login',
          response_model=UserRes,
          response_model_exclude_unset=True)
async def login(req: UserReq, ok: bool = Depends(val_captcha)):
  if not ok:
    raise HTTPException(BAD_REQ, CAPTCHA_ERR)
  docs = await user_col.where('username', '==', req.uname).limit(1).get()
  if len(docs) == 0:
    raise HTTPException(BAD_REQ, USER_OR_PASS_ERR)
  doc: dict = docs[0].to_dict()
  tok = jwt.encode({'exp': datetime.utcnow() + timedelta(days=14)},
                   key=doc['key'])
  return UserRes(uname=req.uname, token=tok)


async def verify_token(x_token: str = Header(default=None)):
  if x_token is None:
    return False
  # jwt.get_unverified_headers()
  tokens = await user_col.select('token').where('token', '==', x_token).get()
  return len(tokens) == 1


@app.get('/file/{fpath:path}')
async def file_get(fpath: str, token_ok: bool = Depends(verify_token)):
  if not token_ok:
    raise HTTPException(HTTPStatus.UNAUTHORIZED, TOKEN_ERR)
  target_path = os.path.join(data_folder, fpath)
  print(target_path)
  if os.path.isdir(target_path):
    print(os.listdir(target_path))
  return True


@app.get('/play/{fpath:path}')
async def play(fpath: str):
  return fpath


if __name__ == '__main__':
  import uvicorn
  uvicorn.run('server:app')