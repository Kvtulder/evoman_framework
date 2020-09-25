import scipy.stats as st

# mean best individual gains
gain_1_plus = [-10.0,-10.0,14.0,14.0,88.0,-10.0,-10.0,-10.0,-20.0,12.0]
gain_1_comma = [-10.0,90.0,20.0,-10.0,-10.0,-10.0,64.0,-10.0,-10.0,-10.0]

gain_2_comma = [86.0,76.0,76.0,92.0,78.0,86.0,82.0,88.0,82.0,76.0]
gain_2_plus = [76.0,88.0,76.0,82.0,76.0,76.0,88.0,82.0,82.0,80.0]

gain_5_comma = [61.8400000000003,68.56000000000026,76.96000000000022,86.32000000000012,81.76000000000018,88.96000000000011,69.64000000000021,67.96000000000024,84.16000000000015,91.00000000000007]
gain_5_plus = [81.04000000000018,67.7200000000003,57.56000000000015,80.4400000000002,83.32000000000016,77.56000000000022,91.00000000000009,81.40000000000018,69.40000000000025,81.04000000000018]

# runs 2-sided ks- and t-test for all enemies
# (ks test results are unused in the final report)
pKS_enemy1 = st.ks_2samp(gain_1_plus, gain_1_comma)[1]
pT_enemy1 = st.ttest_ind(gain_1_plus, gain_1_comma)[1]
print(pKS_enemy1)
print(pT_enemy1)
print('')

pKS_enemy2 = st.ks_2samp(gain_2_plus, gain_2_comma)[1]
pT_enemy2 = st.ttest_ind(gain_2_plus, gain_2_comma)[1]
print(pKS_enemy2)
print(pT_enemy2)
print('')

pKS_enemy5 = st.ks_2samp(gain_5_plus, gain_5_comma)[1]
pT_enemy5 = st.ttest_ind(gain_5_plus, gain_5_comma)[1]
print(pKS_enemy5)
print(pT_enemy5)
