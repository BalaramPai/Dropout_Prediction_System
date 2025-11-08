import pandas as pd
import numpy as np

N = 2000
np.random.seed(42)

students = pd.DataFrame({
    'student_id':[f'S{1000+i}' for i in range(N)],
    'full_name':[f'Student {i}' for i in range(N)],
    'join_year': np.random.choice([2022,2023,2024], N),
    'gender': np.random.choice(['M','F'], N),
    'socioeconomic_cat': np.random.choice(['low','mid','high'], N),
    'household_income': np.random.randint(10000,200000, N),
    'first_gen': np.random.randint(0,2,N)
})

students.to_csv('dropout_labels.csv', index=False)