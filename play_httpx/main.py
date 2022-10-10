# %%
import httpx, asyncio, time


def main():
  try:
    loop = asyncio.get_running_loop()
    loop.create_task(test())
  except RuntimeError as e:
    asyncio.run(test())
  # loop.run_until_complete(play())
  # loop.create_task(play())


async def test():
  befor = time.perf_counter()
  await asyncio.gather(hello(), hello())
  print(time.perf_counter() - befor)


async def hello():
  await asyncio.sleep(1)
  print('hello')


# %%
if __name__ == '__main__':
  main()
