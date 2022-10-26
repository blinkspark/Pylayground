from pydantic import BaseModel
from typing import Union

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