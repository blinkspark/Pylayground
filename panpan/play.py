from httpx import AsyncClient, Headers
import asyncio
import sys,os


# token
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2Njg1MDAzMzAsImlkIjoianRxbnVtdmJ0OTUydGVyIiwidHlwZSI6ImFkbWluIn0.8hM6tfehxITw9iJ3lYPubk2YP8f7K4Z3CAekKsoVbX4
async def main():
  # await login()
  print(os.environ)
  async with AsyncClient() as c:
    res = await c.get(
        'http://127.0.0.1:8090/api/collections',
        headers={
            'Authorization':
            'Admin eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2Njg1MDAzMzAsImlkIjoianRxbnVtdmJ0OTUydGVyIiwidHlwZSI6ImFkbWluIn0.8hM6tfehxITw9iJ3lYPubk2YP8f7K4Z3CAekKsoVbX4'
        })
    print(res.content, res.encoding)


async def login():
  async with AsyncClient() as c:
    res = await c.post('http://127.0.0.1:8090/api/admins/auth-via-email',
                       json={
                           
                       })
    print(res.content)


asyncio.run(main())
