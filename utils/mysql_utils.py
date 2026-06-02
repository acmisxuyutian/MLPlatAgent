import json

import mysql.connector
from mysql.connector import Error

class MySQLDatabase:
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        """建立到MySQL数据库的连接"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            print("Connection to MySQL DB successful")
        except:
            raise Error

    def close_connection(self):
        """关闭数据库连接"""
        if self.connection.is_connected():
            self.connection.close()
            print("MySQL connection is closed")

    # 获取数据库中所有表的信息
    def get_database_info(self):

        query = """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = %s;
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (self.database,))
        results = cursor.fetchall()
        cursor.close()

        database_info = {}
        for row in  results:
            table_name = row[0]
            column_name = row[1]
            if table_name not in database_info:
                database_info[table_name] = []
            database_info[table_name].append(column_name)
        # 遍历字典的每个键值对
        new_database_info = []
        for dataset_name, columns in database_info.items():
            new_database_info.append({
                'dataset_name': dataset_name,
                'columns': columns
            })

        return new_database_info

    def execute_query(self, query):
        """执行SQL查询"""
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully")
        except Error as e:
            print(f"The error '{e}' occurred")

    def read_query(self, query):
        """执行SQL查询并返回结果"""
        cursor = self.connection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            column_names = [i[0] for i in cursor.description]
            return result, column_names
        except Error as e:
            print(f"The error '{e}' occurred")
        finally:
            cursor.close()

if __name__ == '__main__':
    import pandas as pd

    # 使用MySQLDatabase类
    from config import MySQL_Config
    db = MySQLDatabase(host=MySQL_Config["server"], port=MySQL_Config["port"], user=MySQL_Config["username"], password=MySQL_Config["password"], database=MySQL_Config["database"])
    db.connect()

    # 获取并打印所有表的信息
    database_info = db.get_database_info()
    print(len(database_info))
    for d in database_info:
        print(d["dataset_name"])

    # 关闭连接
    db.close_connection()
