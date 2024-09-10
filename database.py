import mysql.connector
from tkinter import messagebox

class Database:
    def __init__(self):
        try:
            self.mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="Khanhminh04-12",
                database="users",
                port=3306,
            )
            self.mycursor = self.mydb.cursor()
        except mysql.connector.Error as err:
            messagebox.showerror('Database Error', f"Error connecting to MySQL: {err}")

    def insert_user(self, user_id, name, age):
        insert_query = "INSERT INTO dang_ki (id, name, age) VALUES (%s, %s, %s)"
        val = (user_id, name, age)
        self.mycursor.execute(insert_query, val)
        self.mydb.commit()

    def get_name_by_id(self, user_id):
        self.mycursor.execute(f"SELECT name FROM dang_ki WHERE id = {user_id}")
        result = self.mycursor.fetchone()
        if result:
            return result[0]
        return None
