import numpy as np
import random
import pandas as pd

dtypes = {"sentence": np.str, "token": np.str, "complexity": np.float64}

# citeste datele de antrenare si testare
train = pd.read_excel('train.xlsx', dtype=dtypes, keep_default_na=False)
test = pd.read_excel('test.xlsx', dtype=dtypes, keep_default_na=False)
print(train.iloc[4323]['sentence'])
# afisam dimensiunile variabilelor citite
print('train data: ', train.shape)
print('test data: ', test.shape)

test_id = np.arange(7663, 9001)
predictions = np.ones(1338)

for i in range(1338):
    predictions[i] = random.randint(0, 1)

np.savetxt("submisie_Kaggle_random.csv", np.stack((test_id, predictions)).T, fmt="%d", delimiter=',',
           header="id,complex", comments="")
