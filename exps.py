import data_process
import pandas as pd
import utils
import numpy as np
from collections import defaultdict
from scipy.stats import gaussian_kde
from obesity_calculations import ObesityCal
from collections import Counter

results_saving_path = 'visual_files/data/'
bmi_cats_labels = ['Underweight', 'Normal', 'Overweight-I', 'Overweight-II', 'Obesity-I', 'Obesity-II', 'Obesity-III',
                   'All']


def add_mortality_info(mortality_info, info, group, death_data_curr, death_data_elig, confidence):
    mortality_info['Info'].append(info)
    mortality_info['Group'].append(group)
    death_info_curr = utils.mean_confidence_interval(death_data_curr, confidence=confidence)
    death_info_elig = utils.mean_confidence_interval(death_data_elig, confidence=confidence)
    mortality_info['Current'].append(str(round(death_info_curr[0])) + ' [' +
                                     str(round(death_info_curr[1])) + ' – ' +
                                     str(round(death_info_curr[2])) + ']')
    mortality_info['Eligible'].append(str(round(death_info_elig[0])) + ' [' +
                                      str(round(death_info_elig[1])) + ' – ' +
                                      str(round(death_info_elig[2])) + ']')


def curr_obe_death_weight_by_age(obesity_cal: ObesityCal):
    age_groups_dict = {'18–64': ['18–29', '30–49', '50–64'],
                       '65 and over': ['65 and over']}
    age_groups_dict_uptake = {'18–29': ['18 years', '19 years', '20–29'],
                              '30–49': ['30–39', '40–49'],
                              '50–64': ['50–59', '60–64'],
                              '65 and over': ['65–69', '70–79', '80 and over']}
    obesity_cal.init_drug_info('0', 'sema')
    curr_obesity_death_weight_by_age = {}
    obe_death_weight_all = 0
    # ------- percent_under_65 and percent_over_65 ---------
    for age_group in age_groups_dict:
        obe_death_weight = 0
        for age in age_groups_dict[age_group]:
            obe_uptake = obesity_cal.obesity_uptake_among_age[age]
            for detailed_age in age_groups_dict_uptake[age]:
                if detailed_age in ['60–64', '65–69']:
                    detailed_age_percent = obesity_cal.age_distribution_list[obesity_cal.age_groups.index('60–69')] / 2
                else:
                    detailed_age_percent = obesity_cal.age_distribution_list[obesity_cal.age_groups.index(detailed_age)]
                obe_death_weight += obe_uptake * detailed_age_percent
        curr_obesity_death_weight_by_age[age_group] = obe_death_weight
        obe_death_weight_all += obe_death_weight

    for age_group in curr_obesity_death_weight_by_age:
        curr_obesity_death_weight_by_age[age_group] /= obe_death_weight_all
    return curr_obesity_death_weight_by_age


def sum_death_by_age(death_info):
    age_groups_dict = {'18–64': ['18 years', '19 years', '20–29', '30–39', '40–49', '50–59', '60–64'],
                       '65 and over': ['65–69', '70–79', '80 and over']}
    death_by_age_all = {}
    for age in age_groups_dict:
        death_by_age = np.zeros(len(death_info['18 years'])) * 1.0
        for detailed_age in age_groups_dict[age]:
            if detailed_age in ['60–64', '65–69']:
                death_by_age += np.array(death_info['60–69']) / 2
            else:
                death_by_age += np.array(death_info[detailed_age])
        death_by_age_all[age] = death_by_age
    return death_by_age_all


def distribute_curr_obe_death_under65_by_insurance(curr_death_under_65, share_of_private):
    curr_death_under_65 = np.array(curr_death_under_65)
    return {'Medicaid': curr_death_under_65 * (1 - share_of_private),
            'Private': curr_death_under_65 * share_of_private,
            'Uninsured': np.zeros(len(curr_death_under_65)) * 1.0}


def total_death_share_under65_by_insurance(obesity_cal):
    dia_curr_access = {'Medicaid': 1, 'Private': 1, 'Uninsured': 0}
    insurance_eligibility_info_df = insurance_eligibility_summary(obesity_cal)
    obe_death_weight_by_insurance = {}
    obe_death_weight_all = 0
    dia_death_weight_by_insurance = {}
    dia_death_weight_all = 0
    dia_curr_death_weight_by_insurance = {}
    dia_curr_death_weight_all = 0

    for insurance in ['Medicaid', 'Private', 'Uninsured']:
        obe_death_weight = (insurance_eligibility_info_df.loc['Share of Population'][insurance] *
                            insurance_eligibility_info_df.loc['Eligible-obesity'][insurance])
        obe_death_weight_all += obe_death_weight
        obe_death_weight_by_insurance[insurance] = obe_death_weight

        dia_death_weight = (insurance_eligibility_info_df.loc['Share of Population'][insurance] *
                            insurance_eligibility_info_df.loc['Eligible-diabetes'][insurance])
        dia_death_weight_all += dia_death_weight
        dia_death_weight_by_insurance[insurance] = dia_death_weight

        dia_curr_death_weight = dia_death_weight * dia_curr_access[insurance]
        dia_curr_death_weight_all += dia_curr_death_weight
        dia_curr_death_weight_by_insurance[insurance] = dia_curr_death_weight

    for insurance in dia_death_weight_by_insurance:
        obe_death_weight_by_insurance[insurance] /= obe_death_weight_all
        dia_death_weight_by_insurance[insurance] /= dia_death_weight_all
        dia_curr_death_weight_by_insurance[insurance] /= dia_curr_death_weight_all
    return {'obesity': obe_death_weight_by_insurance,
            'diabetes': dia_death_weight_by_insurance,
            'curr_diabetes': dia_curr_death_weight_by_insurance}


def mortality_info_by_drug(mortality_info, ADe_curr_dia, ADe_curr_obe, ADe_elig_dia, ADe_elig_obe, confidence):
    add_mortality_info(mortality_info, 'Drug', 'Diabetes',
                       ADe_curr_dia['All'], ADe_elig_dia['All'], confidence)
    add_mortality_info(mortality_info, 'Drug', 'Obesity',
                       ADe_curr_obe['All'], ADe_elig_obe['All'], confidence)


def age_specific_mortality_info(age_group, age_ADe_curr_dia, ADe_curr_obe_all, curr_obe_death_share,
                                age_ADe_curr_and_elig_dia, age_ADe_curr_and_elig_obe):
    age_ADe_curr_dia_list = []
    age_ADe_curr_obe_list = np.array(ADe_curr_obe_all) * curr_obe_death_share
    age_ADe_elig_dia_list = []
    age_ADe_elig_obe_list = []
    age_ADe_curr_and_elig_dia_list = []
    age_ADe_curr_and_elig_obe_list = []
    for i in range(len(ADe_curr_obe_all)):
        age_ADe_curr_dia_list.append(age_ADe_curr_dia[age_group][i])
        age_ADe_curr_and_elig_dia_list.append(age_ADe_curr_and_elig_dia[age_group][i])
        age_ADe_curr_and_elig_obe_list.append(age_ADe_curr_and_elig_obe[age_group][i])
        age_ADe_elig_dia_list.append(age_ADe_curr_and_elig_dia_list[-1] - age_ADe_curr_dia_list[-1])
        age_ADe_elig_obe_list.append(age_ADe_curr_and_elig_obe_list[-1] - age_ADe_curr_obe_list[i])
    return (np.array(age_ADe_curr_dia_list),
            age_ADe_curr_obe_list,
            np.array(age_ADe_elig_dia_list),
            np.array(age_ADe_elig_obe_list),
            np.array(age_ADe_curr_and_elig_dia_list),
            np.array(age_ADe_curr_and_elig_obe_list))


def mortality_info_by_age(curr_obesity_death_weight_by_age, mortality_info,
                          ADe_curr_dia, ADe_curr_obe_all,
                          ADe_curr_and_elig_dia, ADe_curr_and_elig_obe,
                          confidence):
    age_ADe_curr_dia = sum_death_by_age(ADe_curr_dia)
    age_ADe_curr_and_elig_dia = sum_death_by_age(ADe_curr_and_elig_dia)
    age_ADe_curr_and_elig_obe = sum_death_by_age(ADe_curr_and_elig_obe)

    ADe_curr_dia_by_age = {}
    ADe_curr_obe_by_age = {}
    ADe_elig_dia_by_age = {}
    ADe_elig_obe_by_age = {}
    ADe_curr_and_elig_dia_by_age = {}
    ADe_curr_and_elig_obe_by_age = {}
    for age_group in curr_obesity_death_weight_by_age:
        # ------ get obesity deaths share (current) ----------------
        curr_obe_death_share = curr_obesity_death_weight_by_age[age_group]

        # ----- age specific results -------------------------------
        (age_ADe_curr_dia_list, age_ADe_curr_obe_list,
         age_ADe_elig_dia_list, age_ADe_elig_obe_list,
         age_ADe_curr_and_elig_dia_list, age_ADe_curr_and_elig_obe_list) = (
            age_specific_mortality_info(age_group, age_ADe_curr_dia, ADe_curr_obe_all, curr_obe_death_share,
                                        age_ADe_curr_and_elig_dia, age_ADe_curr_and_elig_obe))

        # ----- save results ------------------------------------------
        add_mortality_info(mortality_info, 'Age-drug', age_group + '-' + 'Diabetes',
                           age_ADe_curr_dia_list, age_ADe_elig_dia_list, confidence)
        add_mortality_info(mortality_info, 'Age-drug', age_group + '-' + 'Obesity',
                           age_ADe_curr_obe_list, age_ADe_elig_obe_list, confidence)
        add_mortality_info(mortality_info, 'Age', age_group,
                           age_ADe_curr_dia_list + age_ADe_curr_obe_list,
                           age_ADe_elig_dia_list + age_ADe_elig_obe_list, confidence)

        ADe_curr_dia_by_age[age_group] = age_ADe_curr_dia_list
        ADe_curr_obe_by_age[age_group] = age_ADe_curr_obe_list
        ADe_elig_dia_by_age[age_group] = age_ADe_elig_dia_list
        ADe_elig_obe_by_age[age_group] = age_ADe_elig_obe_list
        ADe_curr_and_elig_dia_by_age[age_group] = age_ADe_curr_and_elig_dia_list
        ADe_curr_and_elig_obe_by_age[age_group] = age_ADe_curr_and_elig_obe_list
    return (ADe_curr_dia_by_age, ADe_curr_obe_by_age,
            ADe_elig_dia_by_age, ADe_elig_obe_by_age,
            ADe_curr_and_elig_dia_by_age, ADe_curr_and_elig_obe_by_age)


def mortality_info_by_insurance(weight_by_insurance,
                                mortality_info,
                                ADe_curr_dia_by_age, ADe_curr_obe_by_age, ADe_elig_dia_by_age, ADe_elig_obe_by_age,
                                ADe_curr_and_elig_dia_by_age, ADe_curr_and_elig_obe_by_age, obe_share_of_private,
                                confidence):
    # 18-64, current
    curr_obe_death_by_insurance = distribute_curr_obe_death_under65_by_insurance(ADe_curr_obe_by_age['18–64'],
                                                                                 obe_share_of_private)
    dia_ADe_curr_under65 = ADe_curr_dia_by_age['18–64']

    # 18-64, current and eligible
    dia_ADe_curr_and_elig_under65 = ADe_curr_and_elig_dia_by_age['18–64']
    obe_ADe_curr_and_elig_under65 = ADe_curr_and_elig_obe_by_age['18–64']

    for insurance in ['Private', 'Medicaid', 'Medicare', 'Uninsured']:
        if insurance == 'Medicare':
            # over 65
            add_mortality_info(mortality_info,
                               'Insurance', 'Medicare',
                               ADe_curr_dia_by_age['65 and over'] + ADe_curr_obe_by_age['65 and over'],
                               ADe_elig_dia_by_age['65 and over'] + ADe_elig_obe_by_age['65 and over'],
                               confidence)
        else:
            ADe_curr_by_insurance = curr_obe_death_by_insurance[insurance] + \
                                    dia_ADe_curr_under65 * weight_by_insurance['curr_diabetes'][insurance]
            ADe_elig_by_insurance = (dia_ADe_curr_and_elig_under65 * weight_by_insurance['diabetes'][insurance] +
                                     obe_ADe_curr_and_elig_under65 * weight_by_insurance['obesity'][insurance] -
                                     ADe_curr_by_insurance)
            add_mortality_info(mortality_info, 'Insurance', insurance,
                               ADe_curr_by_insurance, ADe_elig_by_insurance, confidence)


def table1(obesity_cal: ObesityCal, death_sampling_results, confidence, scenario, obe_share_of_private):
    # -------- total deaths averted under current & current + eligible -------
    ADe_curr = death_sampling_results['ADe_curr']
    ADe_elig = death_sampling_results['ADe_elig']
    ADe_curr_dia = death_sampling_results['ADe_curr_dia']
    ADe_curr_obe = death_sampling_results['ADe_curr_obe']
    ADe_elig_dia = death_sampling_results['ADe_elig_dia']
    ADe_elig_obe = death_sampling_results['ADe_elig_obe']
    ADe_curr_and_elig_dia = death_sampling_results['ADe_curr_and_elig_dia']
    ADe_curr_and_elig_obe = death_sampling_results['ADe_curr_and_elig_obe']

    # ----- collect data --------
    mortality_info = defaultdict(list)

    # ----- total ------
    add_mortality_info(mortality_info, 'Age', '18 and over',
                       ADe_curr['All,All'], ADe_elig['All,All'], confidence)

    # ----- by age --------
    curr_obesity_death_weight_by_age = curr_obe_death_weight_by_age(obesity_cal)
    (ADe_curr_dia_by_age, ADe_curr_obe_by_age,
     ADe_elig_dia_by_age, ADe_elig_obe_by_age,
     ADe_curr_and_elig_dia_by_age, ADe_curr_and_elig_obe_by_age) = (
        mortality_info_by_age(curr_obesity_death_weight_by_age, mortality_info,
                              ADe_curr_dia, ADe_curr_obe['All'],
                              ADe_curr_and_elig_dia, ADe_curr_and_elig_obe,
                              confidence))

    # ---- by drugs -----
    mortality_info_by_drug(mortality_info, ADe_curr_dia, ADe_curr_obe, ADe_elig_dia, ADe_elig_obe, confidence)

    # ------ by insurance -----
    weight_by_insurance = total_death_share_under65_by_insurance(obesity_cal)
    mortality_info_by_insurance(weight_by_insurance, mortality_info,
                                ADe_curr_dia_by_age, ADe_curr_obe_by_age,
                                ADe_elig_dia_by_age, ADe_elig_obe_by_age,
                                ADe_curr_and_elig_dia_by_age, ADe_curr_and_elig_obe_by_age,
                                obe_share_of_private, confidence)

    # ------ save results ------
    mortality_info = pd.DataFrame(mortality_info)
    utils.save_dfs(results_saving_path + 'table1' + scenario + '.xlsx',
                   [mortality_info,
                    pd.DataFrame(curr_obesity_death_weight_by_age, index=[0]),
                    pd.DataFrame(weight_by_insurance)],
                   ['mortality',
                    'curr_obe_death_weight_by_age',
                    'weight_by_insurance'], [False, False, True])


def bmi_value_distribution_average(bmi_d, bmi_samples_list):
    bmi_samples_list = np.array(bmi_samples_list)
    density_list = []
    for i in range(len(bmi_samples_list)):
        density_list.append(gaussian_kde(bmi_samples_list[i]).pdf(bmi_d))
    return density_list


def bmi_cat_distribution_average(bmi_distribution_samples):
    average_results = {}
    for age in bmi_distribution_samples:
        average_results[age] = np.mean(np.array(bmi_distribution_samples[age]), axis=0)
        average_results[age] = average_results[age] / sum(average_results[age])
    return average_results


def population_sample_check(obesity_cal: ObesityCal, samples):
    cdf_compare_path = results_saving_path + 'SM_cdf_compare.xlsx'
    # -------- read data -----------------------------------------------
    population_sex_list = samples['samples_sex']
    population_age_list = samples['samples_age']
    population_weight_list = samples['samples_weight']
    population_height_list = samples['samples_height']
    population_BMI_list = samples['samples_no_access_BMI']
    # ---------- analysis ------------------------------------------------
    measures = ['weight', 'height', 'BMI']
    all_population_measure_list = {'weight': population_weight_list,
                                   'height': population_height_list,
                                   'BMI': population_BMI_list}
    full_percentiles = np.array([0, 5, 10, 15, 25, 50, 75, 85, 90, 95, 100])
    actual_mean = defaultdict(dict)
    for measure_index in range(len(measures)):
        measure = measures[measure_index]

        # ----- actual cdf -------------------
        measure_data_df = pd.read_excel('Data/Processed CDC data.xlsx', sheet_name=measure)
        actual_measure_data = {'percentiles': list(full_percentiles * 0.01)}
        for _, row in measure_data_df.iterrows():
            actual_measure_data[row['Sex'] + ', ' + row['Age']] = list(
                utils.full_percentile_info(row, full_percentiles))
            actual_mean[measure][row['Sex'] + ', ' + row['Age']] = row['Mean']
        actual_measure_data = pd.DataFrame(actual_measure_data)
        actual_measure_data.set_index('percentiles')
        if measure_index == 0:
            actual_measure_data.to_excel(cdf_compare_path, sheet_name=measure + '_actual', index=False)
        else:
            with pd.ExcelWriter(cdf_compare_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                actual_measure_data.to_excel(writer, sheet_name=measure + '_actual', index=False)

        # ----- sample cdf -------------------
        population_measure_list = all_population_measure_list[measure]
        sample_mean = defaultdict(list)
        sample_mean_mean = {}
        for (sex, age) in obesity_cal.sex_age_groups:
            sample_percentile_data = defaultdict(list)
            sample_percentile_data_mean = {}
            # get samples for corresponding sex-age group
            for i in range(len(population_sex_list)):
                sample = population_measure_list[i][(population_sex_list[i] == sex) & (population_age_list[i] == age)]
                # percentiles
                for percentile in full_percentiles:
                    sample_percentile_data[percentile * 0.01].append(np.percentile(sample, percentile))
                for percentile in full_percentiles:
                    sample_percentile_data_mean[percentile * 0.01] = np.mean(sample_percentile_data[percentile * 0.01])
                # mean
                sample_mean[sex + ', ' + age].append(np.mean(sample))
            with pd.ExcelWriter(cdf_compare_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                pd.DataFrame(sample_percentile_data_mean, index=[0]).to_excel(writer, sheet_name=measure + '_' + sex[
                    0] + '_' + age, index=False)
        for sex_age_group in sample_mean:
            sample_mean_mean[sex_age_group] = np.mean(sample_mean[sex_age_group])
        sample_mean_mean_df = pd.DataFrame(sample_mean_mean, index=[0])
        with pd.ExcelWriter(cdf_compare_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            sample_mean_mean_df.to_excel(writer, sheet_name=measure + '_sample_mean', index=False)
    actual_mean_df = pd.DataFrame(actual_mean)
    with pd.ExcelWriter(cdf_compare_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        actual_mean_df.to_excel(writer, sheet_name='actual_mean', index=True)

    # ------ sex-age distribution check ------------------
    sex_age_group_distribution_sample = defaultdict(list)

    for i in range(len(population_sex_list)):
        pop_sex_sample = population_sex_list[i]
        pop_age_sample = population_age_list[i]
        sex_age_count = dict(Counter(tuple(zip(pop_sex_sample, pop_age_sample))))
        for sex, age in obesity_cal.sex_age_groups:
            sex_age_group_distribution_sample[(sex, age)].append(sex_age_count[(sex, age)] / len(pop_sex_sample))
    sex_age_group_distribution_check = {}
    not_norm_sum = 0
    for i in range(len(obesity_cal.sex_age_distribution_list)):
        sex, age = obesity_cal.sex_age_groups[i]
        sex_age_group_distribution_check[sex + ', ' + age] = \
            {'Sample mean (not normed)': np.mean(sex_age_group_distribution_sample[(sex, age)]),
             'Actual': obesity_cal.sex_age_distribution_list[i]}
        not_norm_sum += sex_age_group_distribution_check[sex + ', ' + age]['Sample mean (not normed)']

    for sex_age in sex_age_group_distribution_check:
        sex_age_group_distribution_check[sex_age]['Sample mean'] = (
                sex_age_group_distribution_check[sex_age]['Sample mean (not normed)'] / not_norm_sum)
    sex_age_group_distribution_check_df = pd.DataFrame(sex_age_group_distribution_check).transpose()
    with pd.ExcelWriter(cdf_compare_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        sex_age_group_distribution_check_df.to_excel(writer, sheet_name='sex_age_distribution', index=True)


def eligibility_by_bmi(bmi_cats_, bmi_cats_distribution_, eligibility_among_cat, uptake_dia, uptake_obe):
    # given bmi_cat_ (list), get the eligibility info
    uptake_both = uptake_dia
    sum_distribution = sum(bmi_cats_distribution_)
    currently_use_diabetes_drugs = 0
    not_currently_use_diabetes_drugs = 0
    currently_use_obesity_drugs = 0
    not_currently_use_obesity_drugs = 0
    for single_bmi_cat, single_bmi_distribution in zip(bmi_cats_, bmi_cats_distribution_):
        # ------- eligibility ---------
        elig_both = eligibility_among_cat['Both'][single_bmi_cat]
        elig_dia = eligibility_among_cat['Only d2'][single_bmi_cat]
        elig_both_or_d2 = elig_both + elig_dia
        elig_obe = eligibility_among_cat['Only obesity'][single_bmi_cat]

        # ------- use status ------------
        use_dia = elig_both * uptake_both + elig_dia * uptake_dia
        not_use_dia = elig_both_or_d2 - use_dia
        use_obe = elig_obe * uptake_obe
        not_use_obe = elig_obe - use_obe

        # ------ among the total categories ------
        currently_use_diabetes_drugs += single_bmi_distribution * use_dia
        not_currently_use_diabetes_drugs += single_bmi_distribution * not_use_dia
        currently_use_obesity_drugs += single_bmi_distribution * use_obe
        not_currently_use_obesity_drugs += single_bmi_distribution * not_use_obe

    return (currently_use_diabetes_drugs / sum_distribution, not_currently_use_diabetes_drugs / sum_distribution,
            currently_use_obesity_drugs / sum_distribution, not_currently_use_obesity_drugs / sum_distribution)


def uptake_scenario_analysis(obesity_cal, uptake_info):
    mean_uptake_obesity, mean_uptake_diabetes = uptake_info['mean_uptake_obesity'], uptake_info['mean_uptake_diabetes']
    # create data
    uptake_scenarios = {'Overweight': ['Overweight-grade1', 'Overweight-grade2'],
                        'Obesity': ['Obesity-grade1', 'Obesity-grade2', 'Obesity-grade3']}
    current_diabetes = []
    no_diabetes = []
    current_obesity = []
    no_obesity = []

    for merged_cat in uptake_scenarios:
        ud, nud, uo, nuo = eligibility_by_bmi(uptake_scenarios[merged_cat],
                                              [obesity_cal.ecdf_bmi_distribution[cat] for cat in
                                               uptake_scenarios[merged_cat]],
                                              obesity_cal.drug_eligibility, mean_uptake_diabetes, mean_uptake_obesity)
        current_diabetes.append(ud)
        no_diabetes.append(nud)
        current_obesity.append(uo)
        no_obesity.append(nuo)

    current_diabetes = np.array(current_diabetes)
    no_diabetes = np.array(no_diabetes)
    current_obesity = np.array(current_obesity)
    no_obesity = np.array(no_obesity)
    # plot bars in stack manner
    merged_cats = list(uptake_scenarios.keys())

    # ---- save info-------
    uptake_scenario_info = {'current_diabetes': current_diabetes,
                            'no_diabetes': no_diabetes,
                            'current_obesity': current_obesity,
                            'no_obesity': no_obesity}
    uptake_scenario_info = pd.DataFrame(uptake_scenario_info).transpose()
    uptake_scenario_info.columns = merged_cats
    # ------- uptake among eligible -------------
    overweight_prevalence = sum([obesity_cal.ecdf_bmi_distribution[cat_] for cat_ in uptake_scenarios['Overweight']])
    obesity_prevalence = sum([obesity_cal.ecdf_bmi_distribution[cat_] for cat_ in uptake_scenarios['Obesity']])
    total_eligible = (overweight_prevalence * uptake_scenario_info['Overweight'].sum() +
                      obesity_prevalence * uptake_scenario_info['Obesity'].sum())
    if total_eligible != obesity_cal.total_eligible_frac:
        return
    total_uptake_frac = (uptake_scenario_info.loc['current_diabetes', 'Overweight'] * overweight_prevalence +
                         uptake_scenario_info.loc['current_diabetes', 'Obesity'] * obesity_prevalence +
                         uptake_scenario_info.loc['current_obesity', 'Obesity'] * obesity_prevalence)
    uptake_among_eligible = total_uptake_frac / total_eligible
    return uptake_scenario_info, uptake_among_eligible


def fig1(obesity_cal: ObesityCal):
    # -------- eligibility across age -------------------
    _, eligibility_frac_among_age, age_distribution = eligibility_across_ages(obesity_cal)
    age_groups = {'18–29': ['18 years', '19 years', '20–29'],
                  '30–49': ['30–39', '40–49'],
                  '50–64': ['50–59', '60–64'],
                  '18–64': ['18 years', '19 years', '20–29', '30–39', '40–49', '50–59', '60–64'],
                  '65 and over': ['65–69', '70–79', '80 and over']}

    eligible_among_age_merged = defaultdict(float)
    for age in age_groups:
        population_frac_sum = 0
        for detailed_age in age_groups[age]:
            population_frac_sum += age_distribution[detailed_age]
            eligible_among_age_merged[age] += age_distribution[detailed_age] * eligibility_frac_among_age[detailed_age]
        eligible_among_age_merged[age] /= population_frac_sum

    # -------- eligibility across insurance type -------------------
    (_, _, _, eligible_frac_obesity_under65, eligible_frac_diabetes_under65) = (
        eligible_frac_by_insurance_under65(obesity_cal))
    elig_frac_among_insurance = {'Medicare': eligible_among_age_merged['65 and over'],
                                 'Medicaid': eligible_frac_obesity_under65['public'] +
                                             eligible_frac_diabetes_under65['public'],
                                 'Private': eligible_frac_obesity_under65['private'] +
                                            eligible_frac_diabetes_under65['private'],
                                 'Uninsured': eligible_frac_obesity_under65['none'] +
                                              eligible_frac_diabetes_under65['none']}

    # ------ uptake_scenario_info --------
    drug_uptake_info = {'mean_uptake_obesity': sum(obesity_cal.current_uptake_range['obesity']) / 2,
                        'mean_uptake_diabetes': sum(obesity_cal.current_uptake_range['diabetes']) / 2}
    uptake_scenario_info, uptake_among_eligible = uptake_scenario_analysis(obesity_cal,
                                                                           drug_uptake_info)

    # ------ save data -------------------------------
    utils.save_dfs(results_saving_path + 'fig1.xlsx',
                   [
                       pd.DataFrame(obesity_cal.ecdf_bmi_distribution, index=[0]),
                       pd.DataFrame(obesity_cal.drug_eligibility),
                       pd.DataFrame(drug_uptake_info, index=[0]),
                       pd.DataFrame({'total_eligible_frac': obesity_cal.total_eligible_frac}, index=[0]),
                       uptake_scenario_info,
                       pd.DataFrame(elig_frac_among_insurance, index=[0])
                   ],
                   ['bmi distribution', 'eligibility', 'uptake',
                    'total_eligible', 'uptake_scenario_info', 'eligibility among insurance'],
                   [False, True, False, False, True, False])


def fig2(min_bmi, max_bmi, samples_base, sampling_results_base, samples_hypo, sampling_results_hypo):
    # value distribution
    bmi_d = np.linspace(min_bmi, max_bmi, 1000)
    no_access_bmi_density_base = bmi_value_distribution_average(bmi_d, samples_base['samples_no_access_BMI'])
    current_bmi_density_base = bmi_value_distribution_average(bmi_d, samples_base['samples_current_BMI'])
    eligible_bmi_density_base = bmi_value_distribution_average(bmi_d, samples_base['samples_eligible_BMI'])
    eligible_bmi_density_hypo = bmi_value_distribution_average(bmi_d, samples_hypo['samples_eligible_BMI'])
    mean_value = {'x': bmi_d,
                  'mean_no_access': np.mean(np.array(no_access_bmi_density_base), axis=0),
                  'mean_current': np.mean(np.array(current_bmi_density_base), axis=0),
                  'mean_eligible': np.mean(np.array(eligible_bmi_density_base), axis=0),
                  'mean_eligible_hypo': np.mean(np.array(eligible_bmi_density_hypo), axis=0)}
    # cat distribution
    no_access_bmi_cat_distribution = bmi_cat_distribution_average(
        sampling_results_base['Dis_bmi_noAccess_by_age'])
    current_bmi_cat_distribution = bmi_cat_distribution_average(
        sampling_results_base['Dis_bmi_curr_by_age'])
    eligible_bmi_cat_distribution = bmi_cat_distribution_average(
        sampling_results_base['Dis_bmi_elig_by_age'])
    eligible_bmi_cat_distribution_hypo = bmi_cat_distribution_average(
        sampling_results_hypo['Dis_bmi_elig_by_age'])
    # save results
    utils.save_dfs(results_saving_path + 'fig2.xlsx',
                   [
                       pd.DataFrame(mean_value),
                       pd.DataFrame(no_access_bmi_cat_distribution),
                       pd.DataFrame(current_bmi_cat_distribution),
                       pd.DataFrame(eligible_bmi_cat_distribution),
                       pd.DataFrame(eligible_bmi_cat_distribution_hypo)
                   ],
                   ['mean',
                    'no_access_bmi_distribution', 'current_bmi_distribution', 'eligible_bmi_distribution',
                    'eligible_hypo_bmi_distribution'],
                   [False, False, False, False, False])


def eligibility_across_ages(obesity_cal):
    _, mean_diabetes_prevalence, mean_overweight_among_diabetes, mean_obesity_among_diabetes = (
        obesity_cal.ecdf_diabetes_info())
    age_group_bmi_distribution = dict(obesity_cal.ecdf_age_group_bmi_distribution)

    # ----- eligibility among age ------------
    eligibility_frac_among_age = {}
    age_distribution = {}
    for age_group in age_group_bmi_distribution:
        eligibility_frac_among_age[age_group] = 0
        bmi_distribution_this_age = dict(obesity_cal.ecdf_age_group_bmi_distribution[age_group])
        age_distribution[age_group] = obesity_cal.age_distribution_list[obesity_cal.age_groups.index(age_group)]
        _, _, eligibility_frac_among_age[age_group] = obesity_cal.eligible_frac_cal(
            bmi_distribution_this_age['Obesity-grade1'] +
            bmi_distribution_this_age['Obesity-grade2'] +
            bmi_distribution_this_age['Obesity-grade3'],
            mean_diabetes_prevalence,
            mean_overweight_among_diabetes,
            mean_obesity_among_diabetes
        )
        # ----- modify 60-69 --------
        if age_group == '60–69':
            eligibility_frac_among_age['60–64'] = eligibility_frac_among_age['60–69']
            eligibility_frac_among_age['65–69'] = eligibility_frac_among_age['60–69']
            age_distribution['60–64'] = age_distribution['60–69'] / 2
            age_distribution['65–69'] = age_distribution['60–69'] / 2
            del eligibility_frac_among_age['60–69']
            del age_distribution['60–69']
    return eligibility_frac_among_age, age_distribution


def get_moving_results(no_access_bmi: np.array, current_bmi: np.array, eligible_bmi: np.array, obesity_cal: ObesityCal):
    current_moving_results = {}
    eligible_moving_results = {}
    for elig_bmi in obesity_cal.elig_BMI_range:
        elig_bmi_min, elig_bmi_max = obesity_cal.elig_BMI_range[elig_bmi]
        corr_samples = (no_access_bmi >= elig_bmi_min) & (no_access_bmi < elig_bmi_max)
        corr_current_bmi = current_bmi[corr_samples]
        corr_eligible_bmi = eligible_bmi[corr_samples]
        corr_current_moving_results = np.zeros(obesity_cal.num_bmi_cats) * 1.0
        corr_eligible_moving_results = np.zeros(obesity_cal.num_bmi_cats) * 1.0
        for i in range(obesity_cal.num_bmi_cats):
            bmi_min, bmi_max = obesity_cal.BMI_lows[i], obesity_cal.BMI_highs[i]
            corr_current_moving_results[i] = sum((corr_current_bmi >= bmi_min) & (corr_current_bmi < bmi_max))
            corr_eligible_moving_results[i] = sum((corr_eligible_bmi >= bmi_min) & (corr_eligible_bmi < bmi_max))
        current_moving_results[elig_bmi] = corr_current_moving_results / sum(corr_samples)
        eligible_moving_results[elig_bmi] = corr_eligible_moving_results / sum(corr_samples)
    return current_moving_results, eligible_moving_results


def cal_moving_matrix(obesity_cal: ObesityCal, samples, scenario):
    elig_bmi_cat_labels = {'Overweight': 'Overweight',
                           'Overweight-I': 'Overweight-grade1',
                           'Overweight-II': 'Overweight-grade2',
                           'Obesity': 'Obesity',
                           'Obesity-I': 'Obesity-grade1',
                           'Obesity-II': 'Obesity-grade2',
                           'Obesity-III': 'Obesity-grade3'}

    no_access_BMI = samples['samples_no_access_BMI']
    current_BMI = samples['samples_current_BMI']
    eligible_BMI = samples['samples_eligible_BMI']
    num_samples = obesity_cal.sampling_times

    ave_current_moving_results = {}
    ave_eligible_moving_results = {}

    for i in range(num_samples):
        sample_curr_results, sample_elig_results = get_moving_results(no_access_BMI[i], current_BMI[i], eligible_BMI[i],
                                                                      obesity_cal)
        for elig_bmi in sample_curr_results:
            # current
            if elig_bmi not in ave_current_moving_results:
                ave_current_moving_results[elig_bmi] = sample_curr_results[elig_bmi]
            else:
                ave_current_moving_results[elig_bmi] = (ave_current_moving_results[elig_bmi] * i +
                                                        sample_curr_results[elig_bmi]) / (i + 1)
            # eligible
            if elig_bmi not in ave_eligible_moving_results:
                ave_eligible_moving_results[elig_bmi] = sample_elig_results[elig_bmi]
            else:
                ave_eligible_moving_results[elig_bmi] = (ave_eligible_moving_results[elig_bmi] * i +
                                                         sample_elig_results[elig_bmi]) / (i + 1)
    curr_df = pd.DataFrame(ave_current_moving_results).transpose()
    curr_df.columns = obesity_cal.bmi_categorization
    elig_df = pd.DataFrame(ave_eligible_moving_results).transpose()
    elig_df.columns = obesity_cal.bmi_categorization

    # reorder rows
    order_ave_current_moving_results = {}
    order_ave_eligible_moving_results = {}
    for elig_bmi_label in elig_bmi_cat_labels:
        elig_bmi = elig_bmi_cat_labels[elig_bmi_label]
        order_ave_current_moving_results[elig_bmi_label] = np.round(ave_current_moving_results[elig_bmi] * 100, 2)
        order_ave_eligible_moving_results[elig_bmi_label] = np.round(ave_eligible_moving_results[elig_bmi] * 100, 2)

    order_curr_df = pd.DataFrame(order_ave_current_moving_results).transpose()
    order_curr_df.columns = [label + ' (%)' for label in bmi_cats_labels[:-1]]
    order_elig_df = pd.DataFrame(order_ave_eligible_moving_results).transpose()
    order_elig_df.columns = [label + ' (%)' for label in bmi_cats_labels[:-1]]

    utils.save_dfs('visual_files/data/ave_moving_results' + scenario + '.xlsx',
                   [curr_df, elig_df, order_curr_df, order_elig_df],
                   ['current', 'eligible', 'order current', 'order eligible'],
                   [True, True, True, True])


def info_below_65(obesity_cal: ObesityCal):
    # ----- obesity prevalence -------
    obesity_cal.ecdf_age_group_bmi_distribution.loc['Obesity', :] = (
        obesity_cal.ecdf_age_group_bmi_distribution.loc[['Obesity-grade1',
                                                         'Obesity-grade2',
                                                         'Obesity-grade3'], :].sum(axis=0))
    obesity_prevalence = obesity_cal.ecdf_age_group_bmi_distribution.loc['Obesity', :]
    # ------ merged data --------------
    pop_frac_below_65 = 0
    obesity_prevalence_below_65 = 0
    for age, age_frac in zip(obesity_cal.age_groups, obesity_cal.age_distribution_list):
        if int(age[:2]) < 65:
            if age == '60–69':
                pop_frac_below_65 += age_frac / 2
                obesity_prevalence_below_65 += age_frac / 2 * obesity_prevalence.loc[age]
            else:
                pop_frac_below_65 += age_frac
                obesity_prevalence_below_65 += age_frac * obesity_prevalence.loc[age]
    return obesity_cal.P * pop_frac_below_65, obesity_prevalence_below_65 / pop_frac_below_65


def odds_ratio_uninsured(r_public, insurance_frac_dict, r_uninsured_temp):
    public_frac = insurance_frac_dict['public']
    private_frac = insurance_frac_dict['private']
    return r_uninsured_temp / ((public_frac + private_frac) / (r_public * public_frac + private_frac))


def condition_prevalence_among_insurance(odds, insurance_frac_dict, prevalence_among_non_medicare):
    r_public = odds['public vs private']
    r_uninsured_temp = odds['uninsured vs insured']
    r_uninsured = odds_ratio_uninsured(r_public, insurance_frac_dict, r_uninsured_temp)
    r_array = np.array([r_public, 1, r_uninsured])
    prevalence_among_private = prevalence_among_non_medicare / sum(r_array * np.array([insurance_frac_dict['public'],
                                                                                       insurance_frac_dict['private'],
                                                                                       insurance_frac_dict['none']]))
    return {'public': prevalence_among_private * r_array[0],
            'private': prevalence_among_private * r_array[1],
            'none': prevalence_among_private * r_array[2]}


def eligible_frac_among_insurance(obesity_prevalence_among_insurance: dict,
                                  obesity_cal: ObesityCal):
    _, mean_diabetes_p, mean_overweight_among_diabetes, mean_obesity_among_diabetes = obesity_cal.ecdf_diabetes_info()
    diabetes_prevalence_among_insurance = {}
    eligible_frac_obesity = {}
    eligible_frac_diabetes = {}
    for insurance_type in obesity_prevalence_among_insurance:
        obesity_prevalence = obesity_prevalence_among_insurance[insurance_type]
        diabetes_prevalence = mean_diabetes_p
        diabetes_prevalence_among_insurance[insurance_type] = diabetes_prevalence
        eligible_frac_obesity[insurance_type], eligible_frac_diabetes[insurance_type], _ = (
            obesity_cal.eligible_frac_cal(obesity_prevalence, diabetes_prevalence, mean_overweight_among_diabetes,
                                          mean_obesity_among_diabetes))
    return diabetes_prevalence_among_insurance, eligible_frac_obesity, eligible_frac_diabetes


def eligible_frac_by_insurance_under65(obesity_cal: ObesityCal):
    pop_size_non_medicare, obesity_prevalence_non_medicare = info_below_65(obesity_cal)

    insurance_dict_norm, obesity_odds = data_process.get_insurance_data_below_65()

    obesity_prevalence_among_insurance = condition_prevalence_among_insurance(obesity_odds, insurance_dict_norm,
                                                                              obesity_prevalence_non_medicare)
    diabetes_prevalence_among_insurance, eligible_frac_obesity, eligible_frac_diabetes = eligible_frac_among_insurance(
        obesity_prevalence_among_insurance, obesity_cal)
    return (obesity_prevalence_among_insurance, diabetes_prevalence_among_insurance,
            insurance_dict_norm, eligible_frac_obesity, eligible_frac_diabetes)


def run_exp(obesity_cal, willingness_scenario, current_uptake_scenario, drug_weight_loss_scenario, hr_scenario):
    scenario = (willingness_scenario + '_' + str(current_uptake_scenario) + '_' + str(drug_weight_loss_scenario) +
                '_' + hr_scenario)
    samples = obesity_cal.sampling(willingness_scenario=willingness_scenario,
                                   drug_current_uptake_scenario=current_uptake_scenario,
                                   drug_weight_loss_scenario=drug_weight_loss_scenario,
                                   hr_scenario=hr_scenario)
    death_sampling_results = obesity_cal.get_death_summary('samples/' + scenario + '/', samples)
    return scenario, samples, death_sampling_results


def adjust_state_prevalence(non_adjust_prevalence: list, state_pop: list, pop_level_prevalence):
    non_adjust_prevalence = np.array(non_adjust_prevalence)
    state_pop = np.array(state_pop)
    adjust_ratio = pop_level_prevalence / (sum(non_adjust_prevalence * state_pop) / sum(state_pop))
    return adjust_ratio, non_adjust_prevalence * adjust_ratio


def adjust_state_level_bmi_among_diabetes(non_adjust_state_level_bmi_among_diabetes: list,
                                          adjusted_diabetes_prevalence: np.array,
                                          state_pop: list,
                                          pop_level_bmi_among_diabetes):
    non_adjust_state_level_bmi_among_diabetes = np.array(non_adjust_state_level_bmi_among_diabetes)
    state_pop = np.array(state_pop)
    num_non_adjust_bmi_among_diabetes = sum(state_pop * adjusted_diabetes_prevalence *
                                            non_adjust_state_level_bmi_among_diabetes)
    num_adjusted_pop_diabetes = sum(state_pop * adjusted_diabetes_prevalence)
    adjust_ratio = pop_level_bmi_among_diabetes / (num_non_adjust_bmi_among_diabetes / num_adjusted_pop_diabetes)
    return adjust_ratio, non_adjust_state_level_bmi_among_diabetes * adjust_ratio


def eligible_pop_frac(overweight_p, obesity_p, diabetes_p, overweight_among_d_p, obesity_among_d_p):
    d_among_overweight = diabetes_p * overweight_among_d_p / overweight_p
    d_among_obesity = diabetes_p * obesity_among_d_p / obesity_p

    # eligible people: overweight people with diabetes + obesity people
    eligible_p = overweight_p * d_among_overweight + obesity_p
    # take obesity drugs: only obesity people without diabetes
    obesity_drug_p = obesity_p * (1 - d_among_obesity)
    # take diabetes drugs: all overweight people with diabetes + obesity people with diabetes
    diabetes_drug_p = overweight_p * d_among_overweight + obesity_p * d_among_obesity
    return eligible_p, obesity_drug_p, diabetes_drug_p


def get_state_death_weight_adjusted(obesity_cal: ObesityCal,
                                    obesity_averted_death_mean, diabetes_averted_death_mean,
                                    pop_state_year, diabetes_data_year):
    # ------ get ecdf info --------
    frac_diagnosed_d_among_d = obesity_cal.diagnosed_d_among_all
    obesity_prevalence_ecdf, diabetes_prevalence_mean, overweight_among_diabetes_mean, obesity_among_diabetes_mean = (
        obesity_cal.ecdf_diabetes_info())
    # ----- read states data -------
    states_df = pd.read_excel('Data/Raw data/state_names.xlsx')
    states_info = {}
    for _, row in states_df.iterrows():
        states_info[row['NAME']] = row['STUSPS']

    # ------ read population data ------
    pop_by_state_df = pd.read_excel('Data/Raw data/pop_by_state.xlsx', sheet_name='data')
    pop_by_state = {}
    for _, row in pop_by_state_df.iterrows():
        state = row['State'][1:]
        pop_by_state[state] = row[pop_state_year]

    # ----- read bmi prevalence data -------
    bmi_by_state_df = pd.read_excel('Data/Raw data/bmi_distribution_by_state.xlsx', sheet_name='data')
    overweight_by_state = {}
    obesity_by_state = {}
    for _, row in bmi_by_state_df.iterrows():
        state = row['Location']
        if state != 'United States':
            overweight_by_state[state] = row['Overweight (BMI 25.0-29.9)']
            obesity_by_state[state] = row['Obese (BMI 30-39.9)'] + row['Severly Obese (BMI of 40 or Higher)']
    # ----- read diabetes prevalence data -------
    diabetes_by_state, obese_among_diabetes_by_state, overweight_or_obese_among_diabetes_by_state = (
        data_process.get_diabetes_data(list(states_info.keys()), diabetes_data_year))

    # ----- merge_info--------
    available_states = set(list(states_info.keys()))
    for source_data in [pop_by_state, obesity_by_state, diabetes_by_state,
                        obese_among_diabetes_by_state,
                        overweight_or_obese_among_diabetes_by_state]:
        available_states = available_states.intersection(set(list(source_data.keys())))
    available_state_info = defaultdict(dict)
    available_states_list = []
    available_state_pop = []
    available_state_eligible_frac = []
    available_state_obesity_drug_frac = []
    available_state_diabetes_drug_frac = []

    # --- data for adjust -------
    available_state_obesity_prevalence = []
    available_state_diabetes_prevalence = []
    available_state_overweight_among_diabetes = []
    available_state_obesity_among_diabetes = []

    for state in available_states:
        available_state_info[state]['STUSPS'] = states_info[state]
        available_state_info[state]['pop'] = pop_by_state[state]
        available_state_info[state]['overweight_p'] = overweight_by_state[state]
        available_state_info[state]['obesity_p'] = obesity_by_state[state]
        available_state_info[state]['diag_diabetes_p'] = diabetes_by_state[state]['Percentage']
        available_state_info[state]['frac_diagnosed_d_among_d'] = frac_diagnosed_d_among_d
        available_state_info[state]['diabetes_p'] = available_state_info[state][
                                                        'diag_diabetes_p'] / frac_diagnosed_d_among_d
        available_state_info[state]['overweight_or_obe_among_d_p'] = overweight_or_obese_among_diabetes_by_state[state][
            'Percentage']
        available_state_info[state]['obe_among_d_p'] = obese_among_diabetes_by_state[state]['Percentage']
        available_state_info[state]['overweight_among_d_p'] = (
                available_state_info[state]['overweight_or_obe_among_d_p'] -
                available_state_info[state]['obe_among_d_p'])
        eligible_p, obesity_drug_p, diabetes_drug_p = (
            eligible_pop_frac(available_state_info[state]['overweight_p'],
                              available_state_info[state]['obesity_p'],
                              available_state_info[state]['diabetes_p'],
                              available_state_info[state]['overweight_among_d_p'],
                              available_state_info[state]['obe_among_d_p']))
        available_state_info[state]['eligible_pop_frac'] = eligible_p
        available_state_info[state]['obesity_drug_pop_frac'] = obesity_drug_p
        available_state_info[state]['diabetes_drug_pop_frac'] = diabetes_drug_p

        available_states_list.append(state)
        available_state_pop.append(available_state_info[state]['pop'])
        available_state_eligible_frac.append(available_state_info[state]['eligible_pop_frac'])
        available_state_obesity_drug_frac.append(available_state_info[state]['obesity_drug_pop_frac'])
        available_state_diabetes_drug_frac.append(available_state_info[state]['diabetes_drug_pop_frac'])

        # --- save data for adjust -------
        available_state_obesity_prevalence.append(available_state_info[state]['obesity_p'])
        available_state_diabetes_prevalence.append(available_state_info[state]['diabetes_p'])
        available_state_overweight_among_diabetes.append(available_state_info[state]['overweight_among_d_p'])
        available_state_obesity_among_diabetes.append(available_state_info[state]['obe_among_d_p'])

    # ----- adjust data to match national-level data ----------
    obesity_p_adjust_ratio, obesity_p_adjust = adjust_state_prevalence(available_state_obesity_prevalence,
                                                                       available_state_pop,
                                                                       obesity_prevalence_ecdf)
    diabetes_p_adjust_ratio, diabetes_p_adjust = adjust_state_prevalence(available_state_diabetes_prevalence,
                                                                         available_state_pop,
                                                                         diabetes_prevalence_mean)

    overweight_among_diabetes_adjust_ratio, overweight_among_diabetes_adjust = \
        adjust_state_level_bmi_among_diabetes(available_state_overweight_among_diabetes,
                                              diabetes_p_adjust, available_state_pop, overweight_among_diabetes_mean)

    obesity_among_diabetes_adjust_ratio, obesity_among_diabetes_adjust = \
        adjust_state_level_bmi_among_diabetes(available_state_obesity_among_diabetes,
                                              diabetes_p_adjust, available_state_pop, obesity_among_diabetes_mean)

    eligible_pop_frac_adjust = obesity_p_adjust + diabetes_p_adjust * overweight_among_diabetes_adjust
    obesity_drug_pop_frac_adjust = obesity_p_adjust - diabetes_p_adjust * obesity_among_diabetes_adjust
    diabetes_drug_pop_frac_adjust = diabetes_p_adjust * (
            overweight_among_diabetes_adjust + obesity_among_diabetes_adjust)

    eligible_pop_share = np.array(available_state_pop) * np.array(eligible_pop_frac_adjust)
    eligible_pop_share = eligible_pop_share / np.sum(eligible_pop_share)

    obesity_drug_death_weight = np.array(available_state_pop) * np.array(obesity_drug_pop_frac_adjust)
    obesity_drug_death_weight = obesity_drug_death_weight / np.sum(obesity_drug_death_weight)

    diabetes_drug_death_weight = np.array(available_state_pop) * np.array(diabetes_drug_pop_frac_adjust)
    diabetes_drug_death_weight = diabetes_drug_death_weight / np.sum(diabetes_drug_death_weight)

    for i in range(len(available_states_list)):
        state = available_states_list[i]
        available_state_info[state]['% Population'] = (
            utils.save_percentage_str(available_state_info[state]['pop'] / sum(available_state_pop), 2))

        available_state_info[state]['obesity_p_adjust'] = obesity_p_adjust[i]
        available_state_info[state]['obesity_p_adjust_ratio'] = obesity_p_adjust_ratio

        available_state_info[state]['diabetes_p_adjust'] = diabetes_p_adjust[i]
        available_state_info[state]['diabetes_p_adjust_ratio'] = diabetes_p_adjust_ratio

        available_state_info[state]['overweight_among_diabetes_adjust'] = overweight_among_diabetes_adjust[i]
        available_state_info[state]['overweight_among_diabetes_adjust_ratio'] = overweight_among_diabetes_adjust_ratio

        available_state_info[state]['obesity_among_diabetes_adjust'] = obesity_among_diabetes_adjust[i]
        available_state_info[state]['obesity_among_diabetes_adjust_ratio'] = obesity_among_diabetes_adjust_ratio

        available_state_info[state]['eligible_pop_frac (adjust)'] = eligible_pop_frac_adjust[i]
        available_state_info[state]['Eligible (%)'] = utils.save_percentage_str(eligible_pop_frac_adjust[i], 2)
        available_state_info[state]['obesity_drug_pop_frac (adjust)'] = obesity_drug_pop_frac_adjust[i]
        available_state_info[state]['Eligible-obesity (%)'] = utils.save_percentage_str(obesity_drug_pop_frac_adjust[i],
                                                                                        2)
        available_state_info[state]['diabetes_drug_pop_frac (adjust)'] = diabetes_drug_pop_frac_adjust[i]
        available_state_info[state]['Eligible-diabetes (%)'] = utils.save_percentage_str(
            diabetes_drug_pop_frac_adjust[i], 2)

        available_state_info[state]['eligible_pop_share'] = eligible_pop_share[i]
        available_state_info[state]['Share of eligible population (%)'] = (
            utils.save_percentage_str(available_state_info[state]['eligible_pop_share'], 2))
        available_state_info[state]['obesity_weight'] = obesity_drug_death_weight[i]
        available_state_info[state]['diabetes_weight'] = diabetes_drug_death_weight[i]

        available_state_info[state]['obesity_death'] = (obesity_averted_death_mean *
                                                        available_state_info[state]['obesity_weight'])
        available_state_info[state]['diabetes_death'] = (diabetes_averted_death_mean *
                                                         available_state_info[state]['diabetes_weight'])
        available_state_info[state]['total_death'] = (available_state_info[state]['obesity_death'] +
                                                      available_state_info[state]['diabetes_death'])
        available_state_info[state]['Total deaths'] = round(available_state_info[state]['total_death'])
        available_state_info[state]['Total deaths (obesity)'] = round(available_state_info[state]['obesity_death'])
        available_state_info[state]['Total deaths (diabetes)'] = round(available_state_info[state]['diabetes_death'])

        available_state_info[state]['obesity_death_per100000'] = available_state_info[state]['obesity_death'] / \
                                                                 available_state_info[state]['pop'] * 100000
        available_state_info[state]['diabetes_death_per100000'] = available_state_info[state]['diabetes_death'] / \
                                                                  available_state_info[state]['pop'] * 100000
        available_state_info[state]['total_death_per100000'] = (available_state_info[state]['total_death'] /
                                                                available_state_info[state]['pop']) * 100000

        available_state_info[state]['obesity_death_frac'] = available_state_info[state]['obesity_death'] / \
                                                            available_state_info[state]['total_death']
        available_state_info[state]['diabetes_death_frac'] = available_state_info[state]['diabetes_death'] / \
                                                             available_state_info[state]['total_death']

    available_state_info_df = pd.DataFrame(available_state_info).transpose()
    for rank_info in ['pop', 'eligible_pop_frac (adjust)',
                      'eligible_pop_share',
                      'total_death', 'obesity_death', 'diabetes_death',
                      'total_death_per100000', 'obesity_death_per100000', 'diabetes_death_per100000']:
        available_state_info_df[rank_info + '_R'] = available_state_info_df[rank_info].rank(method='dense',
                                                                                            ascending=False)

    available_state_info_df.to_excel(results_saving_path + 'fig3_death_weight.xlsx', sheet_name='detailed')
    with pd.ExcelWriter(results_saving_path + 'fig3_death_weight.xlsx', engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        pd.DataFrame({'obesity averted deaths': obesity_averted_death_mean,
                      'diabetes averted deaths': diabetes_averted_death_mean}, index=[0]).to_excel(writer,
                                                                                                   sheet_name='total',
                                                                                                   index=False)
    SM_state_eligibility_df = available_state_info_df[['% Population',
                                                       'Eligible-obesity (%)',
                                                       'Eligible-diabetes (%)',
                                                       'Eligible (%)',
                                                       'Share of eligible population (%)',
                                                       ]].copy()
    with pd.ExcelWriter(results_saving_path + 'fig3_death_weight.xlsx', engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        SM_state_eligibility_df.to_excel(writer, sheet_name='SM_eligibility', index=True)

    SM_state_info_df = available_state_info_df[['Eligible (%)',
                                                'Total deaths',
                                                'Total deaths (obesity)',
                                                'Total deaths (diabetes)']].copy()
    with pd.ExcelWriter(results_saving_path + 'fig3_death_weight.xlsx', engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        SM_state_info_df.to_excel(writer, sheet_name='SM_deaths', index=True)
    return available_state_info_df


def BMI_info_summary(obesity_cal: ObesityCal):
    BMI_info = defaultdict(list)
    for i in range(len(obesity_cal.bmi_categorization)):
        bmi_cat = obesity_cal.bmi_categorization[i]
        BMI_info['Index (i)'].append(i + 1)
        BMI_info['BMI group'].append(bmi_cats_labels[i])
        BMI_info['BMI range'].append(str(obesity_cal.BMI_lows[i]) + '--' + str(obesity_cal.BMI_highs[i]))
        BMI_info['Prevalence (%)'].append(utils.save_percentage_str(obesity_cal.ecdf_bmi_distribution[bmi_cat], 2))
        BMI_info['Hazard ratio'].append(round(obesity_cal.hr_all_bmi[i], 3))
        mu_all = obesity_cal.cal_mu_all_bmi_no_drug(obesity_cal.ecdf_bmi_distribution, version='dict')
        BMI_info['Share of annual mortality (%)'].append(
            utils.save_percentage_str(
                mu_all[bmi_cat] * obesity_cal.ecdf_bmi_distribution[bmi_cat] / obesity_cal.mu_no_drug, 2))
    BMI_info_df = pd.DataFrame(BMI_info)
    BMI_info_df.to_excel(results_saving_path + 'SM_BMI_info.xlsx', index=False)


def age_gender_summary(obesity_cal: ObesityCal):
    cdf_compare_path = results_saving_path + 'SM_cdf_compare.xlsx'
    cdf_compare_distribution_info = pd.read_excel(cdf_compare_path, sheet_name='sex_age_distribution', index_col=[0])
    # -------- summary ---------------------------
    age_gender_distribution_info = defaultdict(list)
    age_gender_bmi_distribution_info = defaultdict(list)
    for age_group_index in range(len(obesity_cal.age_groups)):
        age_group = obesity_cal.age_groups[age_group_index]
        for gender in ['', 'Female', 'Male']:
            age_gender_distribution_info['Age'].append(age_group)
            age_gender_distribution_info['Gender'].append(gender)

            age_gender_bmi_distribution_info['Age'].append(age_group)
            age_gender_bmi_distribution_info['Gender'].append(gender)
            if gender == '':
                age_gender_distribution_info['% Population'].append(
                    utils.save_percentage_str(obesity_cal.age_distribution_list[age_group_index], 2))
                age_gender_distribution_info['% Population (Monte Carlo samples)'].append('')
            else:
                sex_age_group_index = obesity_cal.sex_age_groups.index((gender, age_group))
                age_gender_distribution_info['% Population'].append(
                    utils.save_percentage_str(obesity_cal.sex_age_distribution_list[sex_age_group_index], 2))
                age_gender_distribution_info['% Population (Monte Carlo samples)'].append(
                    utils.save_percentage_str(
                        cdf_compare_distribution_info.loc[gender + ', ' + age_group]['Sample mean'], 2))

            for i in range(obesity_cal.num_bmi_cats):
                bmi_cat_label = bmi_cats_labels[i]
                bmi_cat = obesity_cal.bmi_categorization[i]
                if gender == '':
                    age_gender_bmi_distribution_info[bmi_cat_label + ' (%)'].append(
                        utils.save_percentage_str(obesity_cal.ecdf_age_group_bmi_distribution.loc[bmi_cat][age_group],
                                                  2))
                else:
                    age_gender_bmi_distribution_info[bmi_cat_label + ' (%)'].append(
                        utils.save_percentage_str(obesity_cal.ecdf_sex_age_group_bmi_distribution.loc[bmi_cat][
                                                      gender + ', ' + age_group], 2))
    age_gender_distribution_info_df = pd.DataFrame(age_gender_distribution_info)
    age_gender_bmi_distribution_info_df = pd.DataFrame(age_gender_bmi_distribution_info)

    utils.save_dfs(results_saving_path + 'SM_age_gender_info.xlsx',
                   [age_gender_distribution_info_df, age_gender_bmi_distribution_info_df],
                   ['distribution', 'bmi_distribution'],
                   [False, False])


def insurance_eligibility_summary(obesity_cal: ObesityCal):
    insurance_eligibility_info = defaultdict(list)
    # ------ under 65 info --------
    (obesity_prevalence_among_insurance, diabetes_prevalence_among_insurance,
     insurance_dict_norm, eligible_frac_obesity, eligible_frac_diabetes) = (
        eligible_frac_by_insurance_under65(obesity_cal))
    # ------- 65 and above ----------
    (elder_distribution, elder_obesity_prevalence, elder_diabetes_prevalence,
     elder_elig_obe_frac, elder_elig_dia_frac, elder_elig_frac) = obesity_cal.elder_65_and_over_obesity()
    # ------- summary ----------------
    for insurance_raw, insurance in [('65 and over', 'Medicare'),
                                     ('public', 'Medicaid'),
                                     ('private', 'Private'),
                                     ('none', 'Uninsured')]:
        insurance_eligibility_info['Insurance category'].append(insurance)
        if insurance_raw != '65 and over':
            insurance_eligibility_info['Share of Population'].append(
                insurance_dict_norm[insurance_raw] * (1 - elder_distribution))
            insurance_eligibility_info['Obesity prevalence'].append(
                obesity_prevalence_among_insurance[insurance_raw])
            insurance_eligibility_info['Diabetes prevalence'].append(
                diabetes_prevalence_among_insurance[insurance_raw])
            insurance_eligibility_info['Eligible-obesity'].append(
                eligible_frac_obesity[insurance_raw])
            insurance_eligibility_info['Eligible-diabetes'].append(
                eligible_frac_diabetes[insurance_raw])
            insurance_eligibility_info['Eligible'].append(
                (eligible_frac_obesity[insurance_raw] + eligible_frac_diabetes[insurance_raw]))
            insurance_eligibility_info['Share of eligible population'].append(
                obesity_cal.share_of_eligible_pop(
                    eligible_frac_obesity[insurance_raw] + eligible_frac_diabetes[insurance_raw],
                    insurance_dict_norm[insurance_raw] * (1 - elder_distribution)))

        else:
            insurance_eligibility_info['Share of Population'].append(elder_distribution)
            insurance_eligibility_info['Obesity prevalence'].append(elder_obesity_prevalence)
            insurance_eligibility_info['Diabetes prevalence'].append(elder_diabetes_prevalence)
            insurance_eligibility_info['Eligible-obesity'].append(elder_elig_obe_frac)
            insurance_eligibility_info['Eligible-diabetes'].append(elder_elig_dia_frac)
            insurance_eligibility_info['Eligible'].append(elder_elig_frac)
            insurance_eligibility_info['Share of eligible population'].append(
                obesity_cal.share_of_eligible_pop(elder_elig_frac, elder_distribution))
    insurance_eligibility_info_df = pd.DataFrame(insurance_eligibility_info).transpose()
    insurance_eligibility_info_df.to_excel(results_saving_path + 'SM_insurance_elig_info.xlsx',
                                           header='Insurance category')
    insurance_eligibility_info_df.columns = insurance_eligibility_info['Insurance category']
    return insurance_eligibility_info_df
