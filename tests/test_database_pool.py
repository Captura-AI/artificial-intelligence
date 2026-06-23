import pytest

from app.db import database


class FakeConnection:
    def __init__(self) -> None:
        self.rolled_back = False
        self.committed = False

    def rollback(self) -> None:
        self.rolled_back = True

    def commit(self) -> None:
        self.committed = True


class FakePool:
    """Stand-in for ThreadedConnectionPool that tracks borrow/return."""

    def __init__(self) -> None:
        self.conn = FakeConnection()
        self.getconn_calls = 0
        self.putconn_calls = 0
        self.returned_conn = None

    def getconn(self) -> FakeConnection:
        self.getconn_calls += 1
        return self.conn

    def putconn(self, conn) -> None:
        self.putconn_calls += 1
        self.returned_conn = conn


@pytest.fixture
def fake_pool(monkeypatch) -> FakePool:
    pool = FakePool()
    monkeypatch.setattr(database, "_get_pool", lambda *a, **k: pool)
    return pool


def test_connection_borrowed_and_returned(fake_pool):
    with database.get_connection("postgresql://x") as conn:
        assert conn is fake_pool.conn

    assert fake_pool.getconn_calls == 1
    assert fake_pool.putconn_calls == 1
    assert fake_pool.returned_conn is fake_pool.conn
    # Clean exit must not roll back.
    assert fake_pool.conn.rolled_back is False


def test_connection_rolled_back_and_returned_on_error(fake_pool):
    with pytest.raises(ValueError):
        with database.get_connection("postgresql://x"):
            raise ValueError("boom")

    # The aborted transaction is cleared and the connection still returned.
    assert fake_pool.conn.rolled_back is True
    assert fake_pool.putconn_calls == 1
    assert fake_pool.returned_conn is fake_pool.conn
