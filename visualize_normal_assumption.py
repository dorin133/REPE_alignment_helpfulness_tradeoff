import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

with open("/home/binyaminr/Documents/llama-2-13b_high_school_computer_science_race.pkl", 'rb') as f:
    computer_science_data = pickle.load(f)
with open("/home/binyaminr/Documents/llama-2-13b_international_law_race.pkl", 'rb') as f:
    international_law_data = pickle.load(f)
with open("/home/binyaminr/Documents/llama-2-13b_medical_genetics_race.pkl", 'rb') as f:
    medical_genetics_data = pickle.load(f)

for i in range(1, 6):
    curr = np.concatenate([international_law_data[i], computer_science_data[i], medical_genetics_data[i]])
    sns.kdeplot(curr, label=r'$r_e$' + '=' + str(i))

plt.legend()
plt.xlabel(r'$<\delta_{r_e}(q),U^T(e_i - e_{correct})>$')
plt.title('logit delta - harmfulness')
plt.show()

stds_international_law = []
stds_computer_science = []
stds_medical_genetics = []
for i in range(1, 10):
    stds_international_law.append(np.std(international_law_data[i]))
    stds_computer_science.append(np.std(computer_science_data[i]))
    stds_medical_genetics.append(np.std(medical_genetics_data[i]))

plt.figure()
plt.plot(list(range(1, 10)), stds_international_law, label='international_law')
plt.plot(list(range(1, 10)), stds_computer_science, label='computer_science')
plt.plot(list(range(1, 10)), stds_medical_genetics, label='medical_genetics')
plt.legend()
plt.title('std vs REPE coefficient - harmfulness')
plt.xlabel(r'$r_e$')
plt.ylabel('standard deviation')

plt.show()
