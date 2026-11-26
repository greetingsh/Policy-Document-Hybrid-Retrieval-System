# config.py
DB_HOST = "127.0.0.1"
DB_PORT = 5432
DB_NAME = "policy_search_db"
DB_USER = "postgres"
DB_PASS = "wh20040427"

DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
