import numpy as np
from obesity_calculations import ObesityCal
import exps

# -------------- input data --------------------------------------------
mu_no_drug = 984.1 / 100000  # https://www.cdc.gov/nchs/fastats/deaths.htm [2022]
demographics_year = 2022
diabetes_data_year = 2021
confidence = 0.95
obe_share_of_private = 0.8  # https://www.cnbc.com/2024/02/16/what-you-need-to-know-about-health-coverage-for-new-weight-loss-drugs.html

# -------------- init  --------------------------------------------
obesity_cal = ObesityCal(mu_no_drug=mu_no_drug, demographics_year=demographics_year)

# ------------ run experiments 1:baseline ---------------------------------------
main_scenario, main_samples, main_death_sampling_results = exps.run_exp(obesity_cal,
                                                                        willingness_scenario='base',
                                                                        current_uptake_scenario='0',
                                                                        drug_weight_loss_scenario='sema',
                                                                        hr_scenario='white')

# ------------ run experiments 2:optimistic ---------------------------------------
hypo_scenario, hypo_samples, hypo_death_sampling_results = exps.run_exp(obesity_cal,
                                                                        willingness_scenario='hypo',
                                                                        current_uptake_scenario='0',
                                                                        drug_weight_loss_scenario='sema',
                                                                        hr_scenario='white')


# --------------- main text figures & tables ------------------------------
# fig 1
exps.fig1(obesity_cal)
# fig 2
exps.fig2(obesity_cal.min_bmi, obesity_cal.max_bmi,
          main_samples, main_death_sampling_results,
          hypo_samples, hypo_death_sampling_results)
# table 1
exps.table1(obesity_cal, main_death_sampling_results, confidence=confidence, scenario='',
            obe_share_of_private=obe_share_of_private)
# figures 1 and 3
available_state_info_df = (
    exps.get_state_death_weight_adjusted(obesity_cal,
                                         obesity_averted_death_mean=np.mean(
                                             main_death_sampling_results['ADe_elig_obe']['All']),
                                         diabetes_averted_death_mean=np.mean(
                                             main_death_sampling_results['ADe_elig_dia']['All']),
                                         pop_state_year=demographics_year,
                                         diabetes_data_year=diabetes_data_year))


# # --------------- SM figures & tables ------------------------------
# cdf compare
exps.population_sample_check(obesity_cal, main_samples)
# BMI info
exps.BMI_info_summary(obesity_cal)
# age gender info
exps.age_gender_summary(obesity_cal)
# moving matrix
exps.cal_moving_matrix(obesity_cal, main_samples, scenario='')
exps.cal_moving_matrix(obesity_cal, hypo_samples, scenario='_hypo')
