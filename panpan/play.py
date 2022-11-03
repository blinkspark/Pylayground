from httpx import AsyncClient as HttpxAsyncClient
import asyncio
import sys, os, json
from google.cloud.firestore import AsyncClient, DocumentReference
from google.oauth2.service_account import Credentials


async def main():
  venvv = os.environ.get('VENVV')
  print(venvv)
  pass

async def play_firestore():
  c = AsyncClient.from_service_account_json('account.json')
  col = c.collection('users')
  
  docs = [i async for i in col.stream()]
  print(len(docs))
  # res = col.add({'key': '123456', 'password': '321456', 'username': 'testaa'})
  # print(res)
  # for doc in col.list_documents():
  #   print(doc.get().to_dict())


async def pocketbase():
  async with HttpxAsyncClient() as c:

    res = await c.get(
        'http://127.0.0.1:8090/api/collections',
        headers={
            'Authorization':
            'Admin eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2Njg1MTAyMjksImlkIjoic3MxYTdnejRsd3Myajh5IiwidHlwZSI6ImFkbWluIn0.1kQSpY7xLoCslnjs5xyXANp-P_20NP_aqLlffrwMbEE'
        })
  #   print(res.content, res.encoding)
  pass


async def pocketbase_login():
  with open('env.json', 'r') as f:
    cfg = json.load(f)
  async with HttpxAsyncClient() as c:
    res = await c.post('http://127.0.0.1:8090/api/admins/auth-via-email',
                       json=cfg)
    print(res.content)


asyncio.run(main())
