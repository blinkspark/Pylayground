from datetime import datetime
from pydantic import BaseModel
from typing import Union


class BaseReq(BaseModel):
  uid: str


class SessionObj(BaseModel):
  capt_id: str
  capt_value: str
  expires: datetime


class CaptchaReq(BaseReq):
  capt_id: str
  capt_value: str


class CaptchaRes(BaseModel):
  capt_id: str
  capt_img: bytes


class UserReq(CaptchaReq):
  uname: str
  passwd: str


class TokenReq(BaseReq):
  token: str