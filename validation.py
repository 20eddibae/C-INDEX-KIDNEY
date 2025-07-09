import pandas as pd

# load your CSV
df = pd.read_csv('c_index_c_angle_results.csv')

# find any rows where c_index is outside [0,1]
bad_c_index = df[(df['c_index'] < 0) | (df['c_index'] > 1)]

# find any rows where either left or right angle is outside [0,90]
bad_left  = df[(df['angle_left_deg']  < 0) | (df['angle_left_deg']  > 90)]
bad_right = df[(df['angle_right_deg'] < 0) | (df['angle_right_deg'] > 90)]

# find any rows where the total c_angle is outside [0,180]
bad_c_angle = df[(df['c_angle_deg'] < 0) | (df['c_angle_deg'] > 180)]

print("c_index out of [0,1]:\n", bad_c_index)
print("\nangle_left out of [0,90]:\n", bad_left)
print("\nangle_right out of [0,90]:\n", bad_right)
print("\nc_angle_deg out of [0,180]:\n", bad_c_angle)
