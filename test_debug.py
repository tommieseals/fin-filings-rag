import sys
sys.path.insert(0, '.')
from app.rag import get_engine
e = get_engine()
result = e.synthesize("What are the risk factors?")
print(result)
