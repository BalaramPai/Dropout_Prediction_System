# generate_synthetic_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
np.random.seed(0)

N = 2000
students = []
for i in range(N):
    sid = f"S{1000+i}"
    income = int(np.random.lognormal(mean=11, sigma=0.6))  # realistic-ish
    socio = np.random.choice(['low','mid','high'], p=[0.25,0.5,0.25])
    first_gen = np.random.choice([0,1], p=[0.8,0.2])
    gender = np.random.choice(['M','F'])
    students.append([sid, f"Student {i}", np.random.choice([2022,2023,2024]), gender, socio, income, first_gen])
students_df = pd.DataFrame(students, columns=['student_id','full_name','join_year','gender','socioeconomic_cat','household_income','first_gen'])
students_df.to_csv('students.csv', index=False)

# academic_records: 4 course records per student with grades correlated to income & first_gen
records = []
courses = ['CSE101','MTH101','PHY101','ENG101']
term_dates = pd.date_range(end=datetime.today(), periods=4, freq='180D').strftime('%Y-%m-%d').tolist()
for sid in students_df['student_id']:
    base = 65 + (students_df.loc[students_df['student_id']==sid,'household_income'].iloc[0]/100000)*10
    for j,course in enumerate(courses):
        grade = max(0, min(100, np.random.normal(loc=base - np.random.randint(0,10), scale=10)))
        credits = 4
        pass_fail = 1 if grade>=40 else 0
        records.append([sid, term_dates[j%len(term_dates)], course, round(grade,2), credits, pass_fail])
pd.DataFrame(records, columns=['student_id','term_date','course_code','grade','credits','pass_fail']).to_csv('academic_records.csv', index=False)

# attendance: generate last 120 days of sessions, present probability depends on socio/income
dates = [ (datetime.today()-timedelta(days=d)).strftime('%Y-%m-%d') for d in range(120,0,-1) ]
att = []
for sid in students_df['student_id']:
    income = students_df.loc[students_df['student_id']==sid,'household_income'].iloc[0]
    # lower income -> somewhat lower attendance
    p = 0.9 - (max(0,100000-income)/200000)
    for dt in dates:
        present = np.random.binomial(1, p)
        session = np.random.choice(['M','F'])
        att.append([sid, dt, int(present), session])
pd.DataFrame(att, columns=['student_id','date','present','session_id']).to_csv('attendance.csv', index=False)

# behavioral
beh = []
for sid in students_df['student_id']:
    events = np.random.poisson(1)
    total_score = 0
    for e in range(events):
        total_score += np.random.randint(1,6)
        beh.append([sid, (datetime.today()-timedelta(days=np.random.randint(1,400))).strftime('%Y-%m-%d'), 'club', np.random.randint(1,6), 'note'])
pd.DataFrame(beh, columns=['student_id','event_date','activity_type','participation_score','notes']).to_csv('behavioral.csv', index=False)

# dropout labels: 5% true dropouts biased toward low attendance & low grades
# compute simple risk score
df_att = pd.read_csv('attendance.csv').groupby('student_id')['present'].mean().reset_index().rename(columns={'present':'att_rate'})
df_ac = pd.read_csv('academic_records.csv').groupby('student_id')['grade'].mean().reset_index().rename(columns={'grade':'avg_grade'})
score = df_att.merge(df_ac, on='student_id')
score['risk'] = (1 - score['att_rate']) + ((50 - score['avg_grade'])/50).clip(lower=0)
score = score.set_index('student_id')
drop_ids = []
for sid in score.index:
    prob = min(0.5, 0.2 + max(0, score.loc[sid,'risk']*0.5))
    if np.random.rand() < prob:
        drop_ids.append(sid)
drop = pd.DataFrame({'student_id': students_df['student_id'], 'label_date': datetime.today().strftime('%Y-%m-%d'),
                     'dropped': students_df['student_id'].isin(drop_ids).astype(int), 'reason': ''})
drop.to_csv('dropout_labels.csv', index=False)

print("Synthetic CSVs generated: students.csv, academic_records.csv, attendance.csv, behavioral.csv, dropout_labels.csv")
