from pydantic import BaseModel
from typing import Union


class BaseObj(BaseModel):
  uid: str


class CaptchaBase(BaseObj):
  capt_id: str

class CaptchaReq(CaptchaBase):
  capt_value: str | None

class CaptchaRes(CaptchaBase):
  capt_img: bytes | None


class UserObj(CaptchaReq):
  uname: str
  passwd: str


class TokenObj(BaseObj):
  token: str