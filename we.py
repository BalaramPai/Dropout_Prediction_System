# import mysql.connector
# import random
# from datetime import datetime, timedelta
#
# # --- Connect to MySQL ---
# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="baha1528",
#     database="MiniProj"
# )
# cursor = conn.cursor()
#
# # --- 1. Students ---
# genders = ['M', 'F']
# socio = ['low', 'mid', 'high']
#
# for i in range(1, 10001):
#     name = f"Student {i}"
#     gender = random.choice(genders)
#     category = random.choice(socio)
#     income = round(random.uniform(30000, 200000), 2)
#     first_gen = random.randint(0, 1)
#
#     cursor.execute(
#         "INSERT INTO students (student_name, gender, socioeconomic_cat, household_income, first_gen) "
#         "VALUES (%s,%s,%s,%s,%s)",
#         (name, gender, category, income, first_gen)
#     )
# conn.commit()
#
# # --- 2. Academic Records ---
# for student_id in range(1, 10001):
#     avg_grade = round(random.uniform(0, 10), 2)
#     total_credits = random.randint(20, 150)
#     pass_rate = round(random.uniform(0.5, 1.0), 2)
#     failed_courses = random.randint(0, 5)
#     grade_trend = round(random.uniform(-1, 1), 2)
#     credits = 3
#     pass_fail = 1 if avg_grade >= 5 else 0
#     grade = int(avg_grade)
#
#     cursor.execute(
#         "INSERT INTO academic_records (student_id, avg_grade, total_credits, pass_rate, failed_courses, grade_trend, credits, pass_fail, grade) "
#         "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
#         (student_id, avg_grade, total_credits, pass_rate, failed_courses, grade_trend, credits, pass_fail, grade)
#     )
# conn.commit()
#
# # --- 3. Attendance (20 days each) ---
# start_date = datetime(2025, 7, 1)
# for student_id in range(1, 10001):
#     for day in range(20):
#         date = (start_date + timedelta(days=day)).strftime('%Y-%m-%d')
#         present = random.randint(0, 1)
#         attendance_rate = round(random.uniform(0.5, 1.0), 2)
#         attendance_count = present
#
#         cursor.execute(
#             "INSERT INTO attendance (student_id, attendance_rate, attendance_count, present, date) "
#             "VALUES (%s,%s,%s,%s,%s)",
#             (student_id, attendance_rate, attendance_count, present, date)
#         )
# conn.commit()
#
# # --- 4. Behavioral ---
# activities = ['Sports', 'Cultural', 'Technical', 'Volunteering']
# for student_id in range(1, 10001):
#     total_participation = random.randint(0, 50)
#     events_attended = random.randint(0, total_participation)
#     activity_type = random.choice(activities)
#     participation_score = round(random.uniform(0, 100), 2)
#
#     cursor.execute(
#         "INSERT INTO behavioral (student_id, total_participation, events_attended, activity_type, participation_score) "
#         "VALUES (%s,%s,%s,%s,%s)",
#         (student_id, total_participation, events_attended, activity_type, participation_score)
#     )
# conn.commit()
#
# # --- 5. Dropout Labels ---
# for student_id in range(1, 10001):
#     dropped = 1 if random.random() < 0.12 else 0
#     cursor.execute(
#         "INSERT INTO dropout_labels (student_id, dropped) VALUES (%s,%s)",
#         (student_id, dropped)
#     )
# conn.commit()
#
# print("All tables populated successfully!")
#
# cursor.close()
# conn.close()


#
#
#
#
# import mysql.connector
# import pandas as pd
# import numpy as np
# import random
# from faker import Faker
# from datetime import datetime, timedelta
#
# fake = Faker()
# n_students = 8000  # number of students to generate
#
# # MySQL connection
# conn = mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='baha1528',
#     database='MiniProj'
# )
# cursor = conn.cursor()
#
# # ------------------ Generate Students ------------------
# students = []
# for _ in range(n_students):
#     student_name = fake.name()
#     gender = random.choice(['M', 'F'])
#     socioeconomic_cat = random.choices(['low','mid','high'], [0.3,0.5,0.2])[0]
#     household_income = round(np.random.randint(30000, 200000),2)
#     first_gen = random.choice([0,1])
#     students.append((student_name, gender, socioeconomic_cat, household_income, first_gen))
#
# cursor.executemany("""
#     INSERT INTO students (student_name, gender, socioeconomic_cat, household_income, first_gen)
#     VALUES (%s,%s,%s,%s,%s)
# """, students)
# conn.commit()
# print(f"{n_students} students inserted.")
#
# # Get inserted student IDs
# cursor.execute("SELECT student_id FROM students ORDER BY student_id DESC LIMIT %s", (n_students,))
# student_ids = [row[0] for row in cursor.fetchall()]
#
# # ------------------ Generate Academic Records ------------------
# academic_records = []
# for student_id in student_ids:
#     avg_grade = round(np.clip(np.random.normal(6, 2), 0, 10), 2)
#     total_credits = random.randint(18, 25)
#     failed_courses = max(0, int(np.random.normal(1,1)))
#     pass_rate = round(max(0, min(1, total_credits - failed_courses)/total_credits),2)
#     grade_trend = round(np.random.normal(0,1),2)
#     grade = int(np.clip(np.random.normal(6,2),0,10))
#     credits = total_credits
#     pass_fail = 1 if grade >= 5 else 0
#     academic_records.append((student_id, avg_grade, total_credits, pass_rate, failed_courses, grade_trend, credits, pass_fail, grade))
#
# cursor.executemany("""
#     INSERT INTO academic_records (student_id, avg_grade, total_credits, pass_rate, failed_courses, grade_trend, credits, pass_fail, grade)
#     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
# """, academic_records)
# conn.commit()
# print("Academic records inserted.")
#
# # ------------------ Generate Attendance ------------------
# attendance_records = []
# for student_id in student_ids:
#     n_days = 30
#     for i in range(n_days):
#         date = datetime(2025,7,1) + timedelta(days=i)
#         present = np.random.binomial(1, np.random.uniform(0.7,1))  # 70-100% chance present
#         attendance_rate = None  # can calculate later
#         attendance_count = None
#         attendance_records.append((student_id, attendance_rate, attendance_count, present, date))
#
# cursor.executemany("""
#     INSERT INTO attendance (student_id, attendance_rate, attendance_count, present, date)
#     VALUES (%s,%s,%s,%s,%s)
# """, attendance_records)
# conn.commit()
# print("Attendance records inserted.")
#
# # ------------------ Generate Behavioral Records ------------------
# behavioral_records = []
# activity_types = ['sports','cultural','tech','volunteering']
# for student_id in student_ids:
#     total_participation = np.random.randint(0,100)
#     events_attended = np.random.randint(0,10)
#     activity_type = random.choice(activity_types)
#     participation_score = round(np.random.uniform(0,10),2)
#     behavioral_records.append((student_id, total_participation, events_attended, activity_type, participation_score))
#
# cursor.executemany("""
#     INSERT INTO behavioral (student_id, total_participation, events_attended, activity_type, participation_score)
#     VALUES (%s,%s,%s,%s,%s)
# """, behavioral_records)
# conn.commit()
# print("Behavioral records inserted.")
#
# # ------------------ Generate Dropout Labels ------------------
# dropout_labels = []
# for student_id in student_ids:
#     # realistic dropout pattern
#     dropped = 1 if (np.random.rand() < 0.1) else 0  # ~10% dropout
#     dropout_labels.append((student_id, dropped))
#
# cursor.executemany("""
#     INSERT INTO dropout_labels (student_id, dropped)
#     VALUES (%s,%s)
# """, dropout_labels)
# conn.commit()
# print("Dropout labels inserted.")
#
# cursor.close()
# conn.close()
# print("Data generation completed.")






import random
from datetime import datetime, timedelta
import mysql.connector

# ---------------- MySQL Connection ---------------- #
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="baha1528",
    database="MiniProj"
)
cursor = conn.cursor()

# ---------------- Configuration ---------------- #
NUM_STUDENTS = 50000
DROPOUT_RATE = 0.12  # ~12% students drop out
ATTENDANCE_DAYS = 60  # number of days to simulate attendance

genders = ['M', 'F']
socio_categories = ['low', 'mid', 'high']

# ---------------- Helpers ---------------- #
def random_date(start_date, days_range):
    return start_date + timedelta(days=random.randint(0, days_range-1))

def insert_student(name, gender, socio, income, first_gen):
    cursor.execute(
        "INSERT INTO students (student_name, gender, socioeconomic_cat, household_income, first_gen) "
        "VALUES (%s, %s, %s, %s, %s)",
        (name, gender, socio, income, first_gen)
    )
    return cursor.lastrowid

def insert_academic(student_id):
    avg_grade = round(random.uniform(4.0, 9.0), 2)
    total_credits = random.randint(18, 30)
    pass_rate = round(random.uniform(0.6, 1.0), 2)
    failed_courses = random.randint(0, 3)
    grade_trend = round(random.uniform(-0.5, 0.5), 2)
    credits = total_credits
    pass_fail = 1 if avg_grade >= 5 else 0
    grade = int(avg_grade)
    cursor.execute(
        "INSERT INTO academic_records (student_id, avg_grade, total_credits, pass_rate, failed_courses, grade_trend, credits, pass_fail, grade) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (student_id, avg_grade, total_credits, pass_rate, failed_courses, grade_trend, credits, pass_fail, grade)
    )

def insert_attendance(student_id):
    start_date = datetime(2025, 7, 1)
    attendance_count = 0
    for _ in range(ATTENDANCE_DAYS):
        present = random.choices([0,1], weights=[0.15,0.85])[0]
        if present:
            attendance_count += 1
        date = random_date(start_date, ATTENDANCE_DAYS)
        cursor.execute(
            "INSERT INTO attendance (student_id, date, present) VALUES (%s, %s, %s)",
            (student_id, date.strftime("%Y-%m-%d"), present)
        )
    attendance_rate = round(attendance_count/ATTENDANCE_DAYS, 2)
    cursor.execute(
        "UPDATE attendance SET attendance_rate=%s, attendance_count=%s WHERE student_id=%s",
        (attendance_rate, attendance_count, student_id)
    )

def insert_behavioral(student_id):
    total_participation = random.randint(0, 20)
    events_attended = random.randint(0, total_participation)
    activity_type = random.choice(['sports', 'cultural', 'tech', 'community'])
    participation_score = round(random.uniform(0, 1), 2)
    cursor.execute(
        "INSERT INTO behavioral (student_id, total_participation, events_attended, activity_type, participation_score) "
        "VALUES (%s,%s,%s,%s,%s)",
        (student_id, total_participation, events_attended, activity_type, participation_score)
    )

def insert_dropout(student_id, pass_fail):
    dropped = 1 if pass_fail == 0 and random.random() < DROPOUT_RATE else 0
    cursor.execute(
        "INSERT INTO dropout_labels (student_id, dropped) VALUES (%s,%s)",
        (student_id, dropped)
    )

# ---------------- Main Loop ---------------- #
for i in range(NUM_STUDENTS):
    name = f"Student_{i+1}"
    gender = random.choice(genders)
    socio = random.choice(socio_categories)
    income = round(random.uniform(20000, 200000),2)
    first_gen = random.choices([0,1], weights=[0.7,0.3])[0]

    student_id = insert_student(name, gender, socio, income, first_gen)
    insert_academic(student_id)
    insert_attendance(student_id)
    insert_behavioral(student_id)

    # get pass_fail from last academic record
    cursor.execute("SELECT pass_fail FROM academic_records WHERE student_id=%s ORDER BY record_id DESC LIMIT 1", (student_id,))
    pass_fail = cursor.fetchone()[0]

    insert_dropout(student_id, pass_fail)

    if (i+1) % 500 == 0:
        print(f"{i+1} students added...")
        conn.commit()

conn.commit()
cursor.close()
conn.close()
print("Done! 10k students and all associated data added.")
