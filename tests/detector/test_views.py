def test_index(client):
  rv = client.get("/")
  assert "ログイン" in rv.data.decode()
  assert "画像新規登録" in rv.data.decode()