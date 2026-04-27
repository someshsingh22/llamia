import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def lc0_client():
    from agents.tools import LcOClient
    c = LcOClient()
    yield c
    c.close()

@pytest.fixture(scope="session")
def analyst():
    from agents.chess_analyst import ChessAnalyst
    return ChessAnalyst()
