import pandas as pd
import numpy as np
import utils
from collections import defaultdict
from collections import Counter


def get_sex_age_distribution(year, if_adults=True):
    raw_sex_age_distribution = pd.read_excel('Data/Raw data/demographics.xlsx',
                                             sheet_name=str(year) + ' sex-age distribution')
    sex_age_info = {}
    all_population_size = 0
    adults_population_size = 0
    for _, row in raw_sex_age_distribution.iterrows():
        # read age
        if type(row['Age']) == type('-'):
            age = int(row['Age'][:-1])  # 100+
        else:
            age = int(row['Age'])

        all_population_size += row['Male % of Population'] + row['Female % of Population']

        # transfer to age categorization in CDC data
        if if_adults:
            if age < 18:
                continue
        else:
            if age < 2:
                continue

        # eligible age groups
        adults_population_size += row['Male % of Population'] + row['Female % of Population']

        if age <= 19:
            sex_age_info[str(age) + ' years'] = {'Male': row['Male % of Population'],
                                                 'Female': row['Female % of Population']}
        else:
            if age >= 80:
                age_category = '80 and over'
            else:
                age_category = age // 10
                age_category = str(age_category * 10) + '–' + str((age_category + 1) * 10 - 1)

            if age_category in sex_age_info:
                sex_age_info[age_category]['Male'] += row['Male % of Population']
                sex_age_info[age_category]['Female'] += row['Female % of Population']
            else:
                sex_age_info[age_category] = {'Male': row['Male % of Population'],
                                              'Female': row['Female % of Population']}
    # get normed data
    sex_age_info_norm = {}
    for age_category in sex_age_info:
        sex_age_info_norm[age_category] = {'Male': sex_age_info[age_category]['Male'] / adults_population_size,
                                           'Female': sex_age_info[age_category][
                                                         'Female'] / adults_population_size}
    return sex_age_info_norm, adults_population_size / all_population_size


def bmi_distribution_ecdf(BMI_cats, BMI_lows, BMI_highs):
    full_percentiles = np.array([0, 5, 10, 15, 25, 50, 75, 85, 90, 95, 100])
    BMI_ecdf_df = pd.read_excel('Data/Processed CDC data.xlsx', sheet_name='BMI')
    min_bmi, max_bmi = 100, 0

    sex_age_group_bmi_percentiles = {}
    for _, row in BMI_ecdf_df.iterrows():
        percentile_data = utils.full_percentile_info(row, full_percentiles)
        sex_age_group_bmi_percentiles[row['Sex'] + ', ' + row['Age']] = \
            {'Sex-age distribution': row['Sex-age distribution'],
             'Percentile_data': percentile_data}
        min_bmi = min(min_bmi, percentile_data[0])
        max_bmi = max(max_bmi, percentile_data[-1])

    sex_age_group_bmi_distribution = defaultdict(dict)
    age_groups = []
    for sex_age in sex_age_group_bmi_percentiles:
        sex, age = sex_age.split(', ')
        if age not in age_groups:
            age_groups.append(age)
        for cat, bmi_low, bmi_high in zip(BMI_cats, BMI_lows, BMI_highs):
            bmi_high_cdf = utils.get_ecdf(full_percentiles,
                                          sex_age_group_bmi_percentiles[sex_age]['Percentile_data'],
                                          bmi_high)
            bmi_low_cdf = utils.get_ecdf(full_percentiles,
                                         sex_age_group_bmi_percentiles[sex_age]['Percentile_data'],
                                         bmi_low)
            sex_age_group_bmi_distribution[sex_age][cat] = bmi_high_cdf - bmi_low_cdf

    age_group_bmi_distribution = defaultdict(dict)
    for age in age_groups:
        for cat in BMI_cats:
            p_male_age = sex_age_group_bmi_percentiles['Male, ' + age]['Sex-age distribution']
            d_male_age = sex_age_group_bmi_distribution['Male, ' + age][cat]
            p_female_age = sex_age_group_bmi_percentiles['Female, ' + age]['Sex-age distribution']
            d_female_age = sex_age_group_bmi_distribution['Female, ' + age][cat]
            age_group_bmi_distribution[age][cat] = (
                    (p_male_age * d_male_age + p_female_age * d_female_age) / (p_male_age + p_female_age))

    bmi_distribution = {}
    for cat in BMI_cats:
        bmi_distribution[cat] = 0.0
        for sex_age in sex_age_group_bmi_distribution:
            bmi_distribution[cat] += (sex_age_group_bmi_percentiles[sex_age]['Sex-age distribution'] *
                                      sex_age_group_bmi_distribution[sex_age][cat])
    return (pd.DataFrame(sex_age_group_bmi_distribution), pd.DataFrame(age_group_bmi_distribution),
            bmi_distribution, min_bmi, max_bmi)


def get_bmi_data(norm_sex_age_distribution):
    raw_BMI_related_data = {}
    processed_BMI_related_data = defaultdict(dict)
    sex_age_groups = []
    age_groups = []
    sex_age_distribution_list = []
    age_distribution_list = []
    full_percentiles = np.array([0, 5, 10, 15, 25, 50, 75, 85, 90, 95, 100])
    for age_group in norm_sex_age_distribution:
        age_groups.append(age_group)
        age_group_percentage = 0
        for sex in norm_sex_age_distribution[age_group]:
            sex_age_groups.append((sex, age_group))
            sex_age_distribution_list.append(norm_sex_age_distribution[age_group][sex])
            age_group_percentage += norm_sex_age_distribution[age_group][sex]
        age_distribution_list.append(age_group_percentage)
    for sheet_name in ['weight', 'height', 'BMI']:
        # read raw data
        raw_BMI_related_data[sheet_name] = pd.read_excel('Data/Raw data/raw CDC data.xlsx', sheet_name=sheet_name)

        # process each raw
        processed_data = []
        for _, row in raw_BMI_related_data[sheet_name].iterrows():
            if row['Age'] in norm_sex_age_distribution:
                # ----- percentiles ----------
                available_percentiles = []
                available_data_points = []
                for i in range(len(full_percentiles)):
                    percentile = full_percentiles[i]
                    corr_col = str(percentile) + 'th'
                    if corr_col in raw_BMI_related_data[sheet_name].columns and type(row[corr_col]) != type('-'):
                        available_percentiles.append(percentile * 0.01)
                        available_data_points.append(row[corr_col])
                percentile_data = utils.linear_data_interpolation(available_percentiles, available_data_points,
                                                                  full_percentiles * 0.01)
                # ----- processed data ----------
                processed_data.append([row['Sex'], row['Age'], norm_sex_age_distribution[row['Age']][row['Sex']],
                                       row['Mean']] + percentile_data)

                processed_BMI_related_data[(row['Sex'], row['Age'])]['Sex-age distribution'] = (
                    norm_sex_age_distribution)[row['Age']][row['Sex']]
                processed_BMI_related_data[(row['Sex'], row['Age'])]['Percentiles'] = full_percentiles
                processed_BMI_related_data[(row['Sex'], row['Age'])][sheet_name] = \
                    {'Mean': row['Mean'], 'Percentile_data': percentile_data}

        # save data
        processed_data_df = pd.DataFrame(processed_data,
                                         columns=['Sex', 'Age', 'Sex-age distribution', 'Mean'] +
                                                 [str(p) + 'th' for p in full_percentiles])
        if sheet_name == 'weight':
            processed_data_df.to_excel('Data/Processed CDC data.xlsx', sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter('Data/Processed CDC data.xlsx', engine='openpyxl', mode='a') as writer:
                processed_data_df.to_excel(writer, sheet_name=sheet_name, index=False)
    return processed_BMI_related_data, age_groups, sex_age_groups, sex_age_distribution_list, age_distribution_list


def get_sex_age_group_samples(samples_num, BMI_related_data):
    weight_samples = {}
    height_samples = {}
    BMI_samples = {}
    for sex_age_group in BMI_related_data:
        samples = utils.generate_sex_age_group_sample(
            BMI_related_data[sex_age_group]['weight']['Percentile_data'],
            BMI_related_data[sex_age_group]['height']['Percentile_data'],
            BMI_related_data[sex_age_group]['BMI']['Percentile_data'],
            samples_num)
        weight_samples[sex_age_group] = np.array(samples['weight'])
        height_samples[sex_age_group] = np.array(samples['height'])
        BMI_samples[sex_age_group] = np.array(samples['BMI'])
    return weight_samples, height_samples, BMI_samples


def get_population_samples(BMI_related_data, sex_age_groups, sex_age_distribution,
                           sex_age_group_sample_size, population_sample_size):
    w_samples, h_samples, B_samples = get_sex_age_group_samples(sex_age_group_sample_size, BMI_related_data)

    # Re-sampling step
    sex_samples = []
    age_samples = []
    weight_samples = []
    height_samples = []
    BMI_samples = []
    sex_age_ids = np.random.choice(range(len(sex_age_groups)), p=sex_age_distribution, size=population_sample_size)
    num_samples_in_each_group = dict(Counter(sex_age_ids))
    for i in range(len(sex_age_groups)):
        sex_age_group = sex_age_groups[i]
        if i not in num_samples_in_each_group:
            num_samples = 0
        else:
            num_samples = num_samples_in_each_group[i]
        selected_pop_ids = np.random.choice(range(sex_age_group_sample_size), size=num_samples)

        sex_samples += [sex_age_group[0]] * num_samples
        age_samples += [sex_age_group[1]] * num_samples
        weight_samples += list(w_samples[sex_age_group][selected_pop_ids])
        height_samples += list(h_samples[sex_age_group][selected_pop_ids])
        BMI_samples += list(B_samples[sex_age_group][selected_pop_ids])
    return (np.array(sex_samples), np.array(age_samples), np.array(weight_samples), np.array(height_samples),
            np.array(BMI_samples))


def get_hr_info_white(bmi_categorization):
    raw_hr_data = pd.read_excel('Data/Raw data/nejm_hr_data.xlsx', sheet_name='raw data')
    raw_hr_info = {}
    for _, row in raw_hr_data.iterrows():
        raw_hr_info[row['BMI category']] = {'BMI_low': row['BMI_low'],
                                            'BMI_high': row['BMI_high'],
                                            'hr_male': row['hr_male'],
                                            'frac_male': row['frac_male_not_norm'] / raw_hr_data[
                                                'frac_male_not_norm'].sum(),
                                            'num_male': row['num_male'],
                                            'hr_female': row['hr_female'],
                                            'frac_female': row['frac_female_not_norm'] / raw_hr_data[
                                                'frac_female_not_norm'].sum(),
                                            'num_female': row['num_female']
                                            }
    merged_BMI_groups = {'Underweight': ['Underweight'],
                         'Normal': ['Normal-non-ref1', 'Normal-non-ref2', 'Normal-ref'],
                         'Overweight-grade1': ['Overweight-grade1'],
                         'Overweight-grade2': ['Overweight-grade2'],
                         'Obesity-grade1': ['Obesity-grade1'],
                         'Obesity-grade2': ['Obesity-grade2'],
                         'Obesity-grade3': ['Obesity-grade3']
                         }
    ref_group = 'Normal'
    merged_BMI_hr = {}
    for combined_BMI_group in merged_BMI_groups:
        non_standard_groups = merged_BMI_groups[combined_BMI_group]

        hr_male = np.array([])
        frac_male = np.array([])
        num_male = np.array([])

        hr_female = np.array([])
        frac_female = np.array([])
        num_female = np.array([])

        for group in non_standard_groups:

            hr_male = np.append(hr_male, raw_hr_info[group]['hr_male'])
            frac_male = np.append(frac_male, raw_hr_info[group]['frac_male'])
            num_male = np.append(num_male, raw_hr_info[group]['num_male'])

            hr_female = np.append(hr_female, raw_hr_info[group]['hr_female'])
            frac_female = np.append(frac_female, raw_hr_info[group]['frac_female'])
            num_female = np.append(num_female, raw_hr_info[group]['num_female'])

        merged_BMI_hr[combined_BMI_group] = \
            {'hr': np.sum(num_male * frac_male * hr_male + num_female * frac_female * hr_female) / (
                 np.sum(num_male * frac_male + num_female * frac_female))}
    merged_BMI_hr1 = {}
    for combined_BMI_group in merged_BMI_groups:
        merged_BMI_hr1[combined_BMI_group] = merged_BMI_hr[combined_BMI_group]['hr'] / merged_BMI_hr[ref_group]['hr']
    hr_all_bmi = []
    for bmi_cat in bmi_categorization:
        hr_all_bmi.append(merged_BMI_hr1[bmi_cat])
    return np.array(hr_all_bmi)


def get_BMI_range(bmi_categorization):
    path = 'Data/Raw data/BMI_ranges.xlsx'
    data = pd.read_excel(path, sheet_name='bmi range')
    BMI_lows, BMI_highs = [], []
    for bmi_cat in bmi_categorization:
        for _, row in data.iterrows():
            if row['BMI category'] == bmi_cat:
                BMI_lows.append(row['BMI_low'])
                BMI_highs.append(row['BMI_high'])

    bmi_range_among_eligible = {}
    data = pd.read_excel(path, sheet_name='eligible range')
    for _, row in data.iterrows():
        bmi_range_among_eligible[row['BMI category']] = [row['BMI_low'], row['BMI_high']]
    return np.array(BMI_lows), np.array(BMI_highs), bmi_range_among_eligible


def get_BMI_distribution_in_diabetes():
    BMI_distribution_in_diabetes_df = pd.read_excel('Data/Raw data/overweight and obesity.xlsx',
                                                    sheet_name='overweight and obesity')
    BMI_distribution_in_diabetes = {}
    for _, row in BMI_distribution_in_diabetes_df.iterrows():
        BMI_distribution_in_diabetes[row['BMI category']] = (row['min'], row['max'])
    diabetes_prevalence_df = pd.read_excel('Data/Raw data/overweight and obesity.xlsx',
                                           sheet_name='diabetes prevalence')
    diabetes_prevalence_min = diabetes_prevalence_df.iloc[0]['prevalence among adults_min']
    diabetes_prevalence_max = diabetes_prevalence_df.iloc[0]['prevalence among adults_max']
    type2_diabetes_among_diabetes_min = diabetes_prevalence_df.iloc[0]['type2 fraction_min']
    type2_diabetes_among_diabetes_max = diabetes_prevalence_df.iloc[0]['type2 fraction_max']
    diagnosed_d_among_all = (diabetes_prevalence_df.iloc[0]['diagnosed among adults'] /
                             diabetes_prevalence_df.iloc[0]['prevalence among adults'])
    return (BMI_distribution_in_diabetes, diabetes_prevalence_min, diabetes_prevalence_max,
            type2_diabetes_among_diabetes_min, type2_diabetes_among_diabetes_max, diagnosed_d_among_all)


def cal_obesity_current_uptake(age_groups, age_distribution_list, obesity_uptake_among_age, total_eligible_frac):
    age_groups_dict = {'18–29': ['18 years', '19 years', '20–29'],
                       '30–49': ['30–39', '40–49'],
                       '50–64': ['50–59', '60–64'],
                       '65 and over': ['65–69', '70–79', '80 and over']}
    obesity_uptake = 0
    for merged_age in obesity_uptake_among_age:
        uptake = obesity_uptake_among_age[merged_age]
        for detailed_age in age_groups_dict[merged_age]:
            if detailed_age in ['60–64', '65–69']:
                obesity_uptake += age_distribution_list[age_groups.index('60–69')] / 2 * uptake
            else:
                obesity_uptake += age_distribution_list[age_groups.index(detailed_age)] * uptake
    return obesity_uptake / total_eligible_frac


def get_drug_info(uptake_scenario, weight_loss_scenario, age_groups, age_distribution_list, total_eligible_frac):
    drug_info_path = 'Data/Raw data/drug info.xlsx'
    # ----- get data for corresponding scenario -------------------------------------------
    diabetes_drug_uptake_df = pd.read_excel(drug_info_path, sheet_name=uptake_scenario + '_uptake_diabetes').iloc[0]
    obesity_drug_uptake_df = pd.read_excel(drug_info_path, sheet_name=uptake_scenario + '_uptake_obesity')
    drug_weight_loss_df1 = pd.read_excel(drug_info_path,
                                         sheet_name=weight_loss_scenario + '_weight loss distribution1')
    drug_weight_loss_df2 = pd.read_excel(drug_info_path,
                                         sheet_name=weight_loss_scenario + '_weight loss distribution2')
    # get current uptake rate for diabetes
    current_uptake_range = {'diabetes': (diabetes_drug_uptake_df['current_uptake_min'],
                                         diabetes_drug_uptake_df['current_uptake_max'])}
    # get current uptake rate for obesity
    obesity_uptake_among_age = {}
    for _, row in obesity_drug_uptake_df.iterrows():
        obesity_uptake_among_age[row['Age']] = row['Uptake']
    current_obesity_uptake = cal_obesity_current_uptake(age_groups, age_distribution_list, obesity_uptake_among_age,
                                                        total_eligible_frac)
    current_uptake_range['obesity'] = (current_obesity_uptake, current_obesity_uptake)

    # get data for weight loss distribution fitting
    weight_loss = []
    percentages = []
    weight_loss_min, weight_loss_max, weight_loss_mean = (drug_weight_loss_df2.iloc[0]['min'],
                                                          drug_weight_loss_df2.iloc[0]['max'],
                                                          drug_weight_loss_df2.iloc[0]['mean'])
    for _, row in drug_weight_loss_df1.iterrows():
        weight_loss.append(row['weight_loss'])
        percentages.append(row['percentage'])

    # get fitting results
    beta_a, beta_b = utils.eff_beta_fit(weight_loss_mean, weight_loss, percentages, weight_loss_min, weight_loss_max)
    weight_loss_info = {'a': beta_a, 'b': beta_b,
                        'range': weight_loss_max - weight_loss_min,
                        'min': weight_loss_min,
                        'weight_loss_source': np.array(weight_loss),
                        'cdf_source': (1 - np.array(percentages)) * 100}

    # reduction in the effectiveness of drugs with type 2 diabetes compared with without d2
    ratio_eff_min = drug_weight_loss_df2.iloc[0]['ratio_eff_min']
    ratio_eff_max = drug_weight_loss_df2.iloc[0]['ratio_eff_max']
    return current_uptake_range, weight_loss_info, ratio_eff_min, ratio_eff_max, obesity_uptake_among_age


def get_total_pop(year):
    demographics = pd.read_excel('Data/Raw data/demographics.xlsx', sheet_name='total pop')
    for _, row in demographics.iterrows():
        if row['year'] == year:
            return row['total population']


def read_diabetes_df(year: int, info: str):
    df = pd.read_excel('Data/Raw data/' + info + '_' + str(year) + '.xlsx', sheet_name='data')
    data_by_state = {}
    for _, row in df.iterrows():
        state = row['State']
        if state != 'Median of States' and type(row['Percentage']) != type('No Data'):
            if state == 'Virgin Islands of the U.S.':
                state = 'United States Virgin Islands'
            data_by_state[state] = row['Percentage'] / 100
    return data_by_state


def impute_diabetes_data_with_old_data(year: int, info: str, states):
    data_by_state = defaultdict(dict)
    data_by_state_year = read_diabetes_df(year, info)
    data_by_state_1_year_before = read_diabetes_df(year - 1, info)
    data_by_state_2_year_before = read_diabetes_df(year - 1, info)
    for state in states:
        if state in data_by_state_year:
            data_by_state[state] = {'Percentage': data_by_state_year[state],
                                    'Year': year}
        else:
            if state in data_by_state_1_year_before:
                data_by_state[state] = {'Percentage': data_by_state_1_year_before[state],
                                        'Year': year - 1}
            elif state in data_by_state_2_year_before:
                data_by_state[state] = {'Percentage': data_by_state_2_year_before[state],
                                        'Year': year - 2}
    data_by_state_df = pd.DataFrame(data_by_state)
    data_by_state_df.to_excel('Data/' + info + '_merge.xlsx')
    return data_by_state


def get_diabetes_data(states, year):
    diabetes_by_state = impute_diabetes_data_with_old_data(year, 'diabetes_prevalence_by_state', states)
    obese_among_diabetes_by_state = impute_diabetes_data_with_old_data(year, 'obese_among_diabetes_by_state',
                                                                       states)
    overweight_or_obese_among_diabetes_by_state = (
        impute_diabetes_data_with_old_data(year, 'overweight_or_obese_diabetes_state', states))
    return diabetes_by_state, obese_among_diabetes_by_state, overweight_or_obese_among_diabetes_by_state


def get_insurance_data_below_65():
    path = 'Data/Raw data/insurance data.xlsx'
    # ----- coverage -------
    insurance_df = pd.read_excel(path, sheet_name='data').iloc[0]
    insurance_dict_raw = {'public': insurance_df['public insurance'],
                          'private': insurance_df['private insurance'],
                          'none': insurance_df['uninsured']}
    insurance_frac_all = sum(list(insurance_dict_raw.values()))
    insurance_dict_norm = {}
    for type_ in insurance_dict_raw:
        insurance_dict_norm[type_] = insurance_dict_raw[type_] / insurance_frac_all
    # ----- obesity odds ratio --------
    obesity_odds_df = pd.read_excel(path, sheet_name='obesity odds ratio').iloc[0]
    obesity_odds = {'public vs private': obesity_odds_df['public insurance vs private'],
                    'uninsured vs insured': obesity_odds_df['uninsured vs insured']}
    return insurance_dict_norm, obesity_odds


def willingness_and_adherence():
    path = 'Data/Raw data/willingness and adherence.xlsx'
    data = pd.read_excel(path, sheet_name='data').iloc[0]
    return (data['base_willingness'], data['base_adherence non diabetes'],
            data['base_adherence diabetes'],
            data['hypo_willingness'], data['hypo_adherence non diabetes'],
            data['hypo_adherence diabetes'])


def healthcare_access(healthcare_access_year):
    path = 'Data/Raw data/healthcare_access.xlsx'
    average_data = pd.read_excel(path, sheet_name=str(healthcare_access_year) + '-average').iloc[0]
    under65_access_data = pd.read_excel(path, sheet_name=str(healthcare_access_year) + '-under65')
    under65_healthcare_usual_access = {}
    for _, row in under65_access_data.iterrows():
        under65_healthcare_usual_access[row['Type']] = row['Data'] * 0.01
    return average_data['Usual'] * 0.01, under65_healthcare_usual_access


def get_income_hr_raw():
    path = 'Data/Raw data/income.xlsx'
    hr_data = pd.read_excel(path, sheet_name='hr data')
    hr_bound, hr_raw = [], []
    for _, row in hr_data.iterrows():
        hr_bound.append((row['low in 2022'], row['high in 2022']))
        hr_raw.append(row['hr'])
    return hr_bound, hr_raw


def get_income_hr(hr_raw_bound, hr_raw, income_low, income_high):
    for i in range(len(hr_raw_bound)):
        if hr_raw_bound[i][0] <= income_low < hr_raw_bound[i][1]:
            income_low_index = i
        if hr_raw_bound[i][0] <= income_high < hr_raw_bound[i][1]:
            income_high_index = i
    if income_low_index == income_high_index:
        return hr_raw[income_low_index]
    else:
        hr_list = []
        income_range_list = []
        for i in range(income_low_index, income_high_index + 1):
            hr_list.append(hr_raw[i])
            income_range_list.append(max(income_low, hr_raw_bound[i][0]) - min(income_high, hr_raw_bound[i][1]))
        return sum(np.array(hr_list) * np.array(income_range_list)) / sum(income_range_list)


def get_income_bmi_info(income_year):
    path = 'Data/Raw data/income.xlsx'
    # ----- get income info for income_year ---------
    income_data = pd.read_excel(path, sheet_name='data-' + str(income_year))
    income_low = []
    income_high = []
    income_percentage = []
    income_overweight_prevalence = []
    income_obesity_prevalence = []
    for _, row in income_data.iterrows():
        income_low.append(row['household income low'])
        income_high.append(row['household income high'])
        income_percentage.append(row['percentage'])
        income_overweight_prevalence.append(row['overweight prevalence'])
        income_obesity_prevalence.append(row['obesity prevalence'])

    # ----- get raw hr data -----
    hr_raw_bound, hr_raw = get_income_hr_raw()
    hr = []
    for i in range(len(income_low)):
        hr.append(get_income_hr(hr_raw_bound, hr_raw, income_low[i], income_high[i]))
    hr = np.array(hr)
    hr = hr / hr[-1]

    income_info = {'low': np.array(income_low),
                   'high': np.array(income_high),
                   'p': np.array(income_percentage) / sum(income_percentage),
                   'overweight_p': np.array(income_overweight_prevalence) * 0.01,
                   'obesity_p': np.array(income_obesity_prevalence) * 0.01,
                   'hr': hr}
    return income_info
