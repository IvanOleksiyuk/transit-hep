import pandas as pd
import matplotlib pyplot as plt
# Replace 'file.csv' with the path to your CSV file
csv_file_path = '/srv/beegfs/scratch/groups/rodem/oliws/radot_rej_3000.csv'

# Load the CSV file
data = pd.read_csv(csv_file_path)

# Display the first few rows of the dataframe
print(data.head())

plt.figure()

plt.savefig("/home/users/o/oleksiyu/WORK/hyperproject/plots/final2/inverse_rej.png")

plt.figure()

plt.savefig("/home/users/o/oleksiyu/WORK/hyperproject/plots/final2/SIC.png")