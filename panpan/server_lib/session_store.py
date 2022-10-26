from . import CaptchaReq
import pickle


class SessionStore():
  session_dict: dict[str, CaptchaReq]
  data_altered: bool

  def __init__(self):
    self.session_dict = dict()

  def set_session(self, host_ip: str, data: CaptchaReq):
    self.session_dict[host_ip] = data

  def get(self, host_ip: str):
    return self.session_dict[host_ip]

  # def save(self, save_path: str):
  #   with open(save_path, 'w') as f:
  #     pickle.dump(self.session_dict, f)
