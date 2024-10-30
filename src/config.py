import os
from pathlib import Path


def get_google_api_key():
    i = 1
    rs = [os.environ.get("GEMINI_API_KEY")]

    while i <= 10:
        rs.append(os.environ.get("GEMINI_API_KEY_" + str(i)))

        i += 1

    return [i for i in rs if i is not None]


GOOGLE_API_KEY_LIST = get_google_api_key()

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

JINA_API_KEY = os.environ.get("JINA_API_KEY")
