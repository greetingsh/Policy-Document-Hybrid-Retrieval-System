# test_pg.py
from sqlalchemy import create_engine, text
from config import DB_URI

engine = create_engine(DB_URI)

