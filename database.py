import sqlite3

conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM attendance")

attendance_records = cursor.fetchall()
for record in attendance_records:
    print(record)

conn.close()