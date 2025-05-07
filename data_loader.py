# data_loader.py
import pandas as pd
import mysql.connector
from config import MYSQL_CONFIG

def load_data():
    conn = mysql.connector.connect(
        host=MYSQL_CONFIG['host'],
        user=MYSQL_CONFIG['user'],
        password=MYSQL_CONFIG['password'],
        database=MYSQL_CONFIG['database']
    )
    query = f"SELECT * FROM {MYSQL_CONFIG['table']}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df
if __name__ == "__main__":
    df = load_data()
    print(df.head())