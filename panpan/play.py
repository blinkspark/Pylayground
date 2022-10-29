from uuid import uuid4


class A:
  a: str

  def __init__(self) -> None:
    print('A')
    self.a = 'A'


class B:
  b: str

  def __init__(self) -> None:
    print('B')
    self.b = 'B'


class AB(A, B):
  pass


ab = AB()
print(ab.a, ab.b)
