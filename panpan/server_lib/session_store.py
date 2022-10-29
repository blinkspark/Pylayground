from . import SessionObj
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Timer, Lock


class SessionStore():
  timer: Timer

  def __init__(self):
    self.session_dict:dict[str, SessionObj] = dict()

  # def gc(self):
  #   for uid, data in self.session_dict:
  #     pass

  def set_session(self, uid: str, data: SessionObj):
    self.session_dict[uid] = data

  def get(self, uid: str):
    return self.session_dict[uid]

  # def save(self, save_path: str):
  #   with open(save_path, 'w') as f:
  #     pickle.dump(self.session_dict, f)
