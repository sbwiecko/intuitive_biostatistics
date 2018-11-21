from statsmodels.stats.power import tt_ind_solve_power

mean_diff, sd_diff = 3, 10
std_effect_size = mean_diff / sd_diff

n = tt_ind_solve_power(effect_size=std_effect_size, alpha=0.05, power=0.8, \
                       ratio=1, alternative='two-sided')

print('Number in *each* group: {:.5f}'.format(n))
