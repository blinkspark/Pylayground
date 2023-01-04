from Cryptodome.Hash import SHA256
from Cryptodome.PublicKey import RSA, ECC
from Cryptodome.Random import get_random_bytes
from Cryptodome.Cipher import PKCS1_OAEP
from Cryptodome.Signature import pkcs1_15, eddsa
from Cryptodome.IO import PEM
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Queue, Pool, Manager
import binascii
import base64, os, time


def w(work_q: Queue, done_q: Queue):
  print('start')
  while done_q.empty():
    rnd_data = get_random_bytes(1024)
    d = SHA256.new(rnd_data).hexdigest()
    if d.startswith('0000'):
      work_q.put_nowait(d)
      break


if __name__ == '__main__':
  cpu_core_num = os.cpu_count()
  with Manager() as m:
    work_q = m.Queue()
    done_q = m.Queue()
    with ProcessPoolExecutor(cpu_core_num) as p:
      for _ in range(cpu_core_num):
        p.submit(w, work_q, done_q)
      d = work_q.get()
      print(d)
      done_q.put(1)
  # with ProcessPoolExecutor(cpu_core_num) as e:
  #   for i in range(cpu_core_num):
  #     e.submit(w, work_q, done_q)
  #   d = work_q.get()
  #   print('got', d)
  #   done_q.put(1)