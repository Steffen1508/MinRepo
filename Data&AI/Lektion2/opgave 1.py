import sqlite3
from sqlite3 import Error

def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")

connection = create_connection("/home/yurrr/Documents/GitHub/MinRepo/Data&AI/Lektion2/sm_app.sqlite")


create_students_table = """ 
CREATE TABLE IF NOT EXISTS students( 
id INTEGER PRIMARY KEY AUTOINCREMENT, 
student_id CHAR(20) NOT NULL UNIQUE, 
name TEXT NOT NULL, 
major TEXT 
); 
"""

create_courses_table = """
CREATE TABLE IF NOT EXISTS courses(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id CHAR(20) NOT NULL UNIQUE,
    course_name TEXT NOT NULL,
    instructor TEXT NOT NULL
);
"""

execute_query(connection, create_students_table)  
execute_query(connection, create_courses_table)

create_students = """
INSERT INTO
    students(student_id,name,major)
VALUES
    ('S001','Emma Jensen','Artificial Intelligence'),
    ('S002','Mads Nielsen','Software Engineering'),
    ('S003','Sofie Hansen','Data Science'),
    ('S004','Oliver Larsen','Cyber Security'),
    ('S005','Clara Pedersen','Machine Learning')
"""

execute_query(connection, create_students)

create_course = """
INSERT INTO
    courses(course_id, course_name, instructor)
VALUES
    ('C101','Introduction to AI','Dr. Thomas Larsen'),
    ('C102','Python for Data Analysis','Prof. Maria Schmidt'),
    ('C103','Database Systems','Dr. Henrik Madsen'),
    ('C104','Machine Learning Basics','Prof. Anna Sørensen'),
    ('C105','Cyber Security Fundamentals','Dr. Lars Kristensen')
"""

execute_query(connection,create_course)















create_enrollments_table = """
CREATE TABLE IF NOT EXISTS enrollments(
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id CHAR(20) NOT NULL,
    course_id CHAR(20) NOT NULL
);
"""


create_enrollments = """
INSERT INTO enrollments (student_id, course_id)
VALUES
    ('S001','C101'),
    ('S001','C104'),
    ('S002','C102'),
    ('S002','C103'),
    ('S003','C102'),
    ('S003','C104'),
    ('S004','C105'),
    ('S004','C103'),
    ('S005','C101'),
    ('S005','C104');
"""


execute_query(connection, create_enrollments_table)

count = execute_read_query(connection, "SELECT COUNT(*) FROM enrollments;")

if count[0][0] == 0:
    execute_query(connection,create_enrollments)

else:
    print("Enrollments findes allerede")



sort = """
SELECT s.name, c.course_name, c.instructor
FROM enrollments e
JOIN students s ON e.student_id = s.student_id
JOIN courses c ON e.course_id = c.course_id
WHERE s.student_id = 'S001';
"""
result = execute_read_query(connection,sort)
print(result)
connection.close()
