import random
import data_process
import numpy as np
from collections import Counter
from collections import defaultdict
import os
import pickle


class ObesityCal:
    def __init__(self, mu_no_drug, demographics_year):
        self.mu_no_drug = mu_no_drug
        self.demographics_year = demographics_year

        # ----------- sampling data --------------
        self.sex_age_group_sample_size = 1000
        self.population_sample_size = 100000
        self.sampling_times = 1000

        # ------ bmi categorization and ranges -----------------------
        self.bmi_categorization = ['Underweight', 'Normal',
                                   'Overweight-grade1', 'Overweight-grade2',
                                   'Obesity-grade1', 'Obesity-grade2', 'Obesity-grade3']
        self.num_bmi_cats = len(self.bmi_categorization)
        self.BMI_lows, self.BMI_highs, self.elig_BMI_range = data_process.get_BMI_range(self.bmi_categorization)

        # ---------- demographics data ----------------------
        sex_age_distribution, self.adults_pop_size_frac = (
            data_process.get_sex_age_distribution(self.demographics_year))
        self.P = round(data_process.get_total_pop(self.demographics_year) * self.adults_pop_size_frac)
        (self.BMI_related_data, self.age_groups, self.sex_age_groups, self.sex_age_distribution_list,
         self.age_distribution_list) = data_process.get_bmi_data(sex_age_distribution)

        # ------- ecdf bmi distribution ----------------------
        (self.ecdf_sex_age_group_bmi_distribution, self.ecdf_age_group_bmi_distribution,
         self.ecdf_bmi_distribution, self.min_bmi, self.max_bmi) = (
            data_process.bmi_distribution_ecdf(self.bmi_categorization, self.BMI_lows, self.BMI_highs))

        # ---------- diabetes prevalence --------------
        (self.BMI_distribution_among_diabetes, self.diabetes_prevalence_min, self.diabetes_prevalence_max,
         self.type2_diabetes_among_diabetes_min, self.type2_diabetes_among_diabetes_max, self.diagnosed_d_among_all) = (
            data_process.get_BMI_distribution_in_diabetes())

        # -------- eligible population ----------
        self.drug_eligibility, self.eligibility_among_bmi, self.total_eligible_frac = self.eligible_pop()

        # -------- init drug info and hr info------
        self.init_drug_info('0', 'sema')
        self.init_hr_info('white')

        # -------- willingness and persistence ----
        self.base_drug_willingness, self.base_adherence_non_d, self.base_adherence_d, \
            self.hypo_drug_willingness, self.hypo_adherence_non_d, self.hypo_adherence_d = (
                data_process.willingness_and_adherence())

    def ecdf_diabetes_info(self):
        obesity_prevalence_ecdf = (self.ecdf_bmi_distribution['Obesity-grade1'] +
                                   self.ecdf_bmi_distribution['Obesity-grade2'] +
                                   self.ecdf_bmi_distribution['Obesity-grade3'])
        diabetes_prevalence_mean = (self.diabetes_prevalence_min + self.diabetes_prevalence_max) / 2
        overweight_among_diabetes_mean = (
                sum(self.BMI_distribution_among_diabetes['Overweight-grade1 & Overweight-grade2']) / 2)
        obesity_among_diabetes_mean = (
                sum(self.BMI_distribution_among_diabetes['Obesity-grade1 & Obesity-grade2']) / 2 +
                sum(self.BMI_distribution_among_diabetes['Obesity-grade3']) / 2)
        return (obesity_prevalence_ecdf, diabetes_prevalence_mean, overweight_among_diabetes_mean,
                obesity_among_diabetes_mean)

    @staticmethod
    def eligible_frac_cal(obesity_p, diabetes_p, overweight_among_d, obesity_among_d):
        elig_obe_frac = obesity_p - diabetes_p * obesity_among_d
        elig_dia_frac = diabetes_p * (overweight_among_d + obesity_among_d)
        elig_frac = obesity_p + diabetes_p * overweight_among_d
        return elig_obe_frac, elig_dia_frac, elig_frac

    def elder_65_and_over_obesity(self):
        _, diabetes_prevalence_mean, overweight_among_diabetes_mean, obesity_among_diabetes_mean = (
            self.ecdf_diabetes_info())
        # return obesity prevalence
        elder_distribution = 0
        elder_obesity_prevalence = 0
        elder_diabetes_prevalence = diabetes_prevalence_mean
        for age in ['60–69', '70–79', '80 and over']:
            if age == '60–69':
                age_distribution = self.age_distribution_list[self.age_groups.index(age)] / 2
            else:
                age_distribution = self.age_distribution_list[self.age_groups.index(age)]
            elder_distribution += age_distribution
            elder_obesity_prevalence += age_distribution * (
                    self.ecdf_age_group_bmi_distribution.loc['Obesity-grade1'][age] +
                    self.ecdf_age_group_bmi_distribution.loc['Obesity-grade2'][age] +
                    self.ecdf_age_group_bmi_distribution.loc['Obesity-grade3'][age])
        elder_obesity_prevalence = elder_obesity_prevalence / elder_distribution
        elder_elig_obe_frac, elder_elig_dia_frac, elder_elig_frac = (
            self.eligible_frac_cal(elder_obesity_prevalence,
                                   elder_diabetes_prevalence,
                                   overweight_among_diabetes_mean,
                                   obesity_among_diabetes_mean))
        return (elder_distribution, elder_obesity_prevalence, elder_diabetes_prevalence,
                elder_elig_obe_frac, elder_elig_dia_frac, elder_elig_frac)

    def share_of_eligible_pop(self, elig_frac, pop_share):
        return elig_frac * pop_share / self.total_eligible_frac

    def eligible_pop(self):
        mean_d2_among_d = (self.type2_diabetes_among_diabetes_min + self.type2_diabetes_among_diabetes_max) / 2
        total_eligible_frac = 0
        eligible_for_both = {}
        eligible_for_obesity = {}
        eligible_for_d2 = {}
        eligibility_among_bmi = {}
        for bmi_cat in self.ecdf_bmi_distribution:
            mean_diabetes_prevalence = self.get_diabetes_prevalence_among_bmi(bmi_cat,
                                                                              self.ecdf_bmi_distribution,
                                                                              random_=False)

            mean_d2 = mean_diabetes_prevalence * mean_d2_among_d
            mean_non_d2_d = mean_diabetes_prevalence * (1 - mean_d2_among_d)
            mean_non_d2 = 1 - mean_d2
            if bmi_cat in ['Underweight', 'Normal']:
                eligible_for_both[bmi_cat] = 0
                eligible_for_obesity[bmi_cat] = 0
                eligible_for_d2[bmi_cat] = 0
                eligibility_among_bmi[bmi_cat] = 0
            elif bmi_cat == 'Overweight-grade1':
                eligible_for_both[bmi_cat] = 0
                eligible_for_obesity[bmi_cat] = 0
                eligible_for_d2[bmi_cat] = mean_d2
                eligibility_among_bmi[bmi_cat] = mean_d2
            elif bmi_cat == 'Overweight-grade2':
                eligible_for_both[bmi_cat] = mean_d2
                eligible_for_obesity[bmi_cat] = mean_non_d2_d
                eligible_for_d2[bmi_cat] = 0
                eligibility_among_bmi[bmi_cat] = mean_d2 + mean_non_d2_d
            else:
                eligible_for_both[bmi_cat] = mean_d2
                eligible_for_obesity[bmi_cat] = mean_non_d2
                eligible_for_d2[bmi_cat] = 0
                eligibility_among_bmi[bmi_cat] = 1
            total_eligible_frac += self.ecdf_bmi_distribution[bmi_cat] * (eligible_for_both[bmi_cat] +
                                                                          eligible_for_obesity[bmi_cat] +
                                                                          eligible_for_d2[bmi_cat])
        return {'Both': eligible_for_both,
                'Only obesity': eligible_for_obesity,
                'Only d2': eligible_for_d2}, eligibility_among_bmi, total_eligible_frac

    def population_info(self):
        self.pop_sex, self.pop_age, self.pop_weight, self.pop_height, self.pop_BMI \
            = data_process.get_population_samples(self.BMI_related_data,
                                                  self.sex_age_groups,
                                                  self.sex_age_distribution_list,
                                                  sex_age_group_sample_size=self.sex_age_group_sample_size,
                                                  population_sample_size=self.population_sample_size)

        # ------ population sample for overweight and obesity -------------------
        self.drug_elig_pop_index = {}
        for bmi_cat in ['Overweight-grade1', 'Overweight-grade2',
                        'Obesity-grade1', 'Obesity-grade2', 'Obesity-grade3']:
            bmi_cat_index = self.bmi_categorization.index(bmi_cat)
            self.drug_elig_pop_index[bmi_cat] = np.arange(len(self.pop_BMI))[
                (self.BMI_lows[bmi_cat_index] <= self.pop_BMI) & (
                        self.pop_BMI < self.BMI_highs[bmi_cat_index])]
        self.no_access_BMI_distribution_info = self.get_bmi_distribution_from_samples(self.pop_BMI)
        self.no_access_bmi_distribution = self.transfer_bmi_info_to_array(self.no_access_BMI_distribution_info)
        self.mu_all_bmi_no_drug = self.cal_mu_all_bmi_no_drug(self.no_access_bmi_distribution)

    def cal_mu_all_bmi_no_drug(self, bmi_distribution_no_drug, version='array'):
        if version == 'array':
            mu_normal_ref_no_drug = self.mu_no_drug / np.sum(self.hr_all_bmi * bmi_distribution_no_drug)
            return self.hr_all_bmi * mu_normal_ref_no_drug
        elif version == 'dict':
            bmi_distribution_no_drug = self.transfer_bmi_info_to_array(bmi_distribution_no_drug)
            mu_normal_ref_no_drug = self.mu_no_drug / np.sum(self.hr_all_bmi * bmi_distribution_no_drug)
            mu_all_bmi_no_drug = {}
            for i in range(self.num_bmi_cats):
                cat = self.bmi_categorization[i]
                mu_all_bmi_no_drug[cat] = self.hr_all_bmi[i] * mu_normal_ref_no_drug
            return mu_all_bmi_no_drug

    def bmi_among_diabetes(self, bmi_cat_merged, random_):
        if bmi_cat_merged in self.BMI_distribution_among_diabetes:
            bmi_share_min = self.BMI_distribution_among_diabetes[bmi_cat_merged][0]
            bmi_share_max = self.BMI_distribution_among_diabetes[bmi_cat_merged][1]
        else:
            bmi_share_min, bmi_share_max = 0, 0
            for bmi_ in self.BMI_distribution_among_diabetes:
                bmi_share_min += self.BMI_distribution_among_diabetes[bmi_][1]
                bmi_share_max += self.BMI_distribution_among_diabetes[bmi_][0]
            bmi_share_min = 1 - bmi_share_min
            bmi_share_max = 1 - bmi_share_max
        if random_:
            return np.random.uniform(bmi_share_min, bmi_share_max)
        else:
            return (bmi_share_min + bmi_share_max) / 2

    def get_diabetes_prevalence_among_bmi(self, bmi_cat, no_access_BMI_distribution_info, random_=True):
        bmi_cats = {'Underweight': 'Underweight & Normal',
                    'Normal': 'Underweight & Normal',
                    'Overweight-grade1': 'Overweight-grade1 & Overweight-grade2',
                    'Overweight-grade2': 'Overweight-grade1 & Overweight-grade2',
                    'Obesity-grade1': 'Obesity-grade1 & Obesity-grade2',
                    'Obesity-grade2': 'Obesity-grade1 & Obesity-grade2',
                    'Obesity-grade3': 'Obesity-grade3'}
        if random_:
            diabetes_prevalence = np.random.uniform(self.diabetes_prevalence_min, self.diabetes_prevalence_max)
        else:
            diabetes_prevalence = (self.diabetes_prevalence_min + self.diabetes_prevalence_max) / 2
        bmi_among_diabetes = self.bmi_among_diabetes(bmi_cats[bmi_cat], random_=random_)
        bmi_distribution = 0
        for bmi_cat_single in bmi_cats[bmi_cat].split(' & '):
            bmi_distribution += no_access_BMI_distribution_info[bmi_cat_single]
        diabetes_prevalence_bmi = diabetes_prevalence * bmi_among_diabetes / bmi_distribution
        return diabetes_prevalence_bmi

    def transfer_bmi_info_to_array(self, bmi_distribution_dict: dict):
        return np.array([bmi_distribution_dict[cat] for cat in self.bmi_categorization])

    def death_num_cal(self, pop_no_access_bmi_distribution: np.array, sub_pop_bmi_distribution: np.array, sub_pop_P):
        mu_by_bmi = self.cal_mu_all_bmi_no_drug(pop_no_access_bmi_distribution)
        death_by_bmi = mu_by_bmi * sub_pop_bmi_distribution * sub_pop_P
        return death_by_bmi, sum(death_by_bmi), death_by_bmi / sum(death_by_bmi)

    def get_bmi_distribution_from_samples(self, sample_bmi, norm=True):
        sample_bmi_cats = np.zeros_like(sample_bmi)
        for i in range(self.num_bmi_cats):
            sample_bmi_cats[(sample_bmi >= self.BMI_lows[i]) & (sample_bmi < self.BMI_highs[i])] = i
        sample_bmi_cats_count = dict(Counter(sample_bmi_cats))
        bmi_prevalence_dict = defaultdict()
        for i in range(self.num_bmi_cats):
            if i not in sample_bmi_cats_count:
                sample_bmi_cats_count[i] = 0
            if norm:
                bmi_prevalence_dict[self.bmi_categorization[i]] = sample_bmi_cats_count[i] / len(sample_bmi)
            else:
                bmi_prevalence_dict[self.bmi_categorization[i]] = sample_bmi_cats_count[i]
        return bmi_prevalence_dict

    def sample_weight_loss(self, total_samples):
        weight_loss_samples = (np.random.beta(a=self.weight_loss_info['a'],
                                              b=self.weight_loss_info['b'],
                                              size=total_samples)
                               * self.weight_loss_info['range']
                               + self.weight_loss_info['min']) / 100
        return weight_loss_samples

    @staticmethod
    def bmi_cal(pop_weight, pop_height):
        """
        :param pop_weight: array of weight [kg]
        :param pop_height: array of height [cm]
        :return: array of bmi
        """
        return pop_weight / ((pop_height * 0.01) ** 2)

    def bmi_count_after_losing_weight(self, pop_index, pop_condition):
        drug_weight_loss = self.sample_weight_loss(len(pop_index))
        if pop_condition == 'd2':
            ratio_eff_d2 = np.random.uniform(self.ratio_eff_min, self.ratio_eff_max, len(pop_index))
        else:
            ratio_eff_d2 = np.ones(len(pop_index))
        drug_weight_loss *= ratio_eff_d2

        new_bmi = self.bmi_cal(self.pop_weight[pop_index] * (1 - drug_weight_loss),
                               self.pop_height[pop_index])
        return list(new_bmi)

    def moving_results_cal(self, drug_pop_with_d2, drug_pop_without_d2):
        index_collection = []
        bmi_collection = []

        # ----- moving due to drug use -------
        for (drug_pop, pop_condition) in zip([drug_pop_with_d2, drug_pop_without_d2], ['d2', 'non-d2']):
            new_bmi_collection = self.bmi_count_after_losing_weight(drug_pop, pop_condition)
            index_collection += list(drug_pop)
            bmi_collection += new_bmi_collection
        return index_collection, bmi_collection

    def sample_drug_uptake(self, bmi_cat,
                           diabetes_drug_uptake_min, diabetes_drug_uptake_max,
                           obesity_drug_uptake_min, obesity_drug_uptake_max):
        pop_index = self.drug_elig_pop_index[bmi_cat]
        # ----------- sample diabetes info ------------------------------------------------
        diabetes_prevalence_among_bmi = self.get_diabetes_prevalence_among_bmi(bmi_cat,
                                                                               self.no_access_BMI_distribution_info)
        type2_diabetes_among_diabetes = np.random.uniform(self.type2_diabetes_among_diabetes_min,
                                                          self.type2_diabetes_among_diabetes_max)
        d2_prevalence_among_bmi = diabetes_prevalence_among_bmi * type2_diabetes_among_diabetes

        # ----------- sample drug uptake rate ---------------------------------------------
        only_d2_uptake = np.random.uniform(diabetes_drug_uptake_min, diabetes_drug_uptake_max)
        only_obesity_uptake = np.random.uniform(obesity_drug_uptake_min, obesity_drug_uptake_max)
        both_uptake = only_d2_uptake

        # ----------- get number of individual taking drugs and losing weight with and without type2 diabetes ------
        if bmi_cat == 'Overweight-grade1':
            drug_pop_with_d2_frac = d2_prevalence_among_bmi * only_d2_uptake
            drug_pop_without_d2_frac = 0
        elif bmi_cat == 'Overweight-grade2':
            drug_pop_with_d2_frac = d2_prevalence_among_bmi * both_uptake
            non_d2_d_prevalence = diabetes_prevalence_among_bmi * (1 - type2_diabetes_among_diabetes)
            drug_pop_without_d2_frac = non_d2_d_prevalence * only_obesity_uptake
        elif bmi_cat in ['Obesity-grade1', 'Obesity-grade2', 'Obesity-grade3']:
            drug_pop_with_d2_frac = d2_prevalence_among_bmi * both_uptake
            non_d2_prevalence = 1 - d2_prevalence_among_bmi
            drug_pop_without_d2_frac = non_d2_prevalence * only_obesity_uptake

        drug_pop_with_d2_num = round(drug_pop_with_d2_frac * len(pop_index))
        drug_pop_without_d2_num = min(round(drug_pop_without_d2_frac * len(pop_index)),
                                      len(pop_index) - drug_pop_with_d2_num)

        # ------------ get index of individual taking drugs and losing weight with and without type2 diabetes --------
        drug_pop_index = random.sample(list(pop_index), drug_pop_with_d2_num + drug_pop_without_d2_num)
        drug_pop_index_with_d2 = drug_pop_index[:drug_pop_with_d2_num]
        drug_pop_index_without_d2 = drug_pop_index[drug_pop_with_d2_num:]

        # -------- return index ---------------------------------------------------
        return drug_pop_index_with_d2, drug_pop_index_without_d2

    def complete_bmi_info(self, change_bmi_index, change_bmi_value):
        sample_BMI = np.array(self.pop_BMI)
        sample_BMI[change_bmi_index] = change_bmi_value
        return sample_BMI

    def complete_drug_info(self, pop_index_taking_dia, pop_index_taking_obe):
        # diabetes drugs:1; obesity drugs:2
        sample_drug = np.zeros_like(self.pop_BMI)
        sample_drug[pop_index_taking_dia] = 1
        sample_drug[pop_index_taking_obe] = 2
        return sample_drug

    def sampling_single(self, drug_willingness, adherence_d, adherence_non_d):
        # ------- get population sample -------------------------------------------
        self.population_info()

        # ------- sampling --------------------------------------------------------
        curr_bmi_index, curr_bmi = [], []
        curr_dia, curr_obe = [], []
        eligible_bmi_index, eligible_bmi = [], []
        eligible_dia, eligible_obe = [], []

        for bmi_cat in self.drug_elig_pop_index:
            # -------- get population taking drugs and losing weight with and without type 2 diabetes ----------
            current_drug_pop_with_d2, current_drug_pop_without_d2 = (
                self.sample_drug_uptake(bmi_cat,
                                        self.current_uptake_range['diabetes'][0] * adherence_d,
                                        self.current_uptake_range['diabetes'][1] * adherence_d,
                                        self.current_uptake_range['obesity'][0] * adherence_non_d,
                                        self.current_uptake_range['obesity'][1] * adherence_non_d))
            eligible_drug_pop_with_d2, eligible_drug_pop_without_d2 = (
                self.sample_drug_uptake(bmi_cat,
                                        drug_willingness * adherence_d,
                                        drug_willingness * adherence_d,
                                        drug_willingness * adherence_non_d,
                                        drug_willingness * adherence_non_d))

            # -------- drug taking info ------
            curr_dia += list(current_drug_pop_with_d2)
            curr_obe += list(current_drug_pop_without_d2)

            eligible_dia += list(eligible_drug_pop_with_d2)
            eligible_obe += list(eligible_drug_pop_without_d2)

            # -------- get current moving results after taking drugs ---------
            curr_index_collection, curr_bmi_collection = self.moving_results_cal(current_drug_pop_with_d2,
                                                                                 current_drug_pop_without_d2)
            # add bmi info
            curr_bmi_index += curr_index_collection
            curr_bmi += curr_bmi_collection

            # -------- get eligible moving results after taking drugs ---------
            eligible_index_collection, eligible_bmi_collection = self.moving_results_cal(eligible_drug_pop_with_d2,
                                                                                         eligible_drug_pop_without_d2)
            # add bmi info
            eligible_bmi_index += eligible_index_collection
            eligible_bmi += eligible_bmi_collection

        return (curr_bmi_index, curr_bmi, curr_dia, curr_obe,
                eligible_bmi_index, eligible_bmi, eligible_dia, eligible_obe)

    def init_drug_info(self, drug_uptake_scenario, drug_weight_loss_scenario):
        (self.current_uptake_range, self.weight_loss_info, self.ratio_eff_min, self.ratio_eff_max,
         self.obesity_uptake_among_age) = (
            data_process.get_drug_info(drug_uptake_scenario, drug_weight_loss_scenario,
                                       self.age_groups,
                                       self.age_distribution_list,
                                       self.total_eligible_frac))

    def init_hr_info(self, hr_scenario):
        if hr_scenario == 'white':
            self.hr_all_bmi = data_process.get_hr_info_white(self.bmi_categorization)

    @staticmethod
    def read_samples(sample_path, willingness_scenario):
        if os.path.exists(sample_path):
            if willingness_scenario == 'base':
                samples = {'samples_sex': np.load(sample_path + 'samples_sex.npy'),
                           'samples_age': np.load(sample_path + 'samples_age.npy'),
                           'samples_weight': np.load(sample_path + 'samples_weight.npy'),
                           'samples_height': np.load(sample_path + 'samples_height.npy'),
                           'samples_no_access_BMI': np.load(sample_path + 'samples_no_access_BMI.npy'),
                           'samples_current_BMI': np.load(sample_path + 'samples_current_BMI.npy'),
                           'samples_current_drug': np.load(sample_path + 'samples_current_drug.npy'),
                           'samples_eligible_BMI': np.load(sample_path + 'samples_eligible_BMI.npy'),
                           'samples_eligible_drug': np.load(sample_path + 'samples_eligible_drug.npy')}
            else:
                samples = {'samples_age': np.load(sample_path + 'samples_age.npy'),
                           'samples_no_access_BMI': np.load(sample_path + 'samples_no_access_BMI.npy'),
                           'samples_current_BMI': np.load(sample_path + 'samples_current_BMI.npy'),
                           'samples_current_drug': np.load(sample_path + 'samples_current_drug.npy'),
                           'samples_eligible_BMI': np.load(sample_path + 'samples_eligible_BMI.npy'),
                           'samples_eligible_drug': np.load(sample_path + 'samples_eligible_drug.npy')}
            return samples
        else:
            os.makedirs(sample_path)
            return None

    def sampling(self, drug_current_uptake_scenario, drug_weight_loss_scenario, willingness_scenario, hr_scenario):
        sample_path = 'samples/' + willingness_scenario + '_' + str(drug_current_uptake_scenario) + '_' + str(
            drug_weight_loss_scenario) + '_' + hr_scenario + '/'
        samples = self.read_samples(sample_path, willingness_scenario)
        if samples:
            return samples
        # ---------- drug info & hr & willingness & adherence based on scenarios-----------
        self.init_drug_info(drug_current_uptake_scenario, drug_weight_loss_scenario)
        self.init_hr_info(hr_scenario)
        if willingness_scenario == 'base':
            drug_willingness, adherence_d, adherence_non_d = (self.base_drug_willingness,
                                                              self.base_adherence_d,
                                                              self.base_adherence_non_d)
        if willingness_scenario == 'hypo':
            drug_willingness, adherence_d, adherence_non_d = (self.hypo_drug_willingness,
                                                              self.hypo_adherence_d,
                                                              self.hypo_adherence_non_d)

        # -------- sampling results -----------------------------------------
        samples_sex = []
        samples_age = []
        samples_weight = []
        samples_height = []
        samples_no_access_BMI = []
        samples_current_BMI = []
        samples_current_drug = []
        samples_eligible_BMI = []
        samples_eligible_drug = []

        for sample_step in range(self.sampling_times):
            # ----- get sample info --------------------------------------
            (sample_curr_bmi_index, sample_curr_bmi, sample_curr_dia, sample_curr_obe,
             sample_eligible_bmi_index, sample_eligible_bmi, sample_eligible_dia, sample_eligible_obe) \
                = self.sampling_single(drug_willingness, adherence_d, adherence_non_d)
            samples_sex.append(self.pop_sex)
            samples_age.append(self.pop_age)
            samples_weight.append(self.pop_weight)
            samples_height.append(self.pop_height)
            samples_no_access_BMI.append(self.pop_BMI)
            samples_current_BMI.append(self.complete_bmi_info(sample_curr_bmi_index, sample_curr_bmi))
            samples_current_drug.append(self.complete_drug_info(sample_curr_dia, sample_curr_obe))
            samples_eligible_BMI.append(self.complete_bmi_info(sample_eligible_bmi_index, sample_eligible_bmi))
            samples_eligible_drug.append(self.complete_drug_info(sample_eligible_dia, sample_eligible_obe))

        # --------- save samples ----------------------
        samples = {'samples_sex': samples_sex,
                   'samples_age': samples_age,
                   'samples_weight': samples_weight,
                   'samples_height': samples_height,
                   'samples_no_access_BMI': samples_no_access_BMI,
                   'samples_current_BMI': samples_current_BMI,
                   'samples_current_drug': samples_current_drug,
                   'samples_eligible_BMI': samples_eligible_BMI,
                   'samples_eligible_drug': samples_eligible_drug
                   }
        if willingness_scenario == 'base':
            np.save(sample_path + 'samples_sex.npy', samples_sex)
            np.save(sample_path + 'samples_weight.npy', samples_weight)
            np.save(sample_path + 'samples_height.npy', samples_height)
        np.save(sample_path + 'samples_age.npy', samples_age)
        np.save(sample_path + 'samples_no_access_BMI.npy', samples_no_access_BMI)
        np.save(sample_path + 'samples_current_BMI.npy', samples_current_BMI)
        np.save(sample_path + 'samples_current_drug.npy', samples_current_drug)
        np.save(sample_path + 'samples_eligible_BMI.npy', samples_eligible_BMI)
        np.save(sample_path + 'samples_eligible_drug.npy', samples_eligible_drug)
        return samples

    def get_death_info_by_age(self, Dis_bmi_noAccess, age_bmi_list, age_pop_list):
        age_Dis_bmi = self.transfer_bmi_info_to_array(self.get_bmi_distribution_from_samples(age_bmi_list))
        age_De_by_bmi, age_De, _ = self.death_num_cal(Dis_bmi_noAccess, age_Dis_bmi, age_pop_list)
        return age_Dis_bmi, age_De_by_bmi, age_De

    def map_bmi_to_mu(self, bmi_array, no_access_mu_all):
        mu_array = np.zeros(len(bmi_array)) * 1.0
        for i in range(self.num_bmi_cats):
            mu_array[(bmi_array >= self.BMI_lows[i]) & (bmi_array < self.BMI_highs[i])] = no_access_mu_all[i]
        return mu_array

    def get_averted_death_by_drug(self, S_drug_info, S_noAccess_bmi, S_withAccess_bmi, no_access_mu_all, age_pop):
        # all compare with no access
        ADe_drug = {}

        S_mu_noAccess = self.map_bmi_to_mu(S_noAccess_bmi, no_access_mu_all)
        S_mu_withAccess = self.map_bmi_to_mu(S_withAccess_bmi, no_access_mu_all)
        S_Amu = S_mu_noAccess - S_mu_withAccess

        for drug in [1, 2]:
            if drug not in S_drug_info:
                ADe_drug[drug] = 0
            else:
                S_drug = S_drug_info == drug
                drug_Amu = S_Amu[S_drug]
                ADe_drug[drug] = sum(drug_Amu) * age_pop / len(S_drug_info)
        return ADe_drug

    def get_death_info(self, S_age, S_noAccess_bmi, S_curr_bmi, S_curr_drug, S_elig_bmi, S_elig_drug):
        # for each sampling result
        S_Dis_bmi_noAccess = self.transfer_bmi_info_to_array(self.get_bmi_distribution_from_samples(S_noAccess_bmi))
        S_no_access_mu_all = self.cal_mu_all_bmi_no_drug(S_Dis_bmi_noAccess)

        # death & averted deaths
        S_De_noAccess = defaultdict(dict)
        S_De_curr = defaultdict(dict)
        S_De_elig = defaultdict(dict)
        S_ADe_curr_dia = {}
        S_ADe_curr_obe = {}
        S_ADe_elig_dia = {}
        S_ADe_elig_obe = {}

        # bmi distribution by age
        S_Dis_bmi_noAccess_by_age = {}
        S_Dis_bmi_curr_by_age = {}
        S_Dis_bmi_elig_by_age = {}

        for age in self.age_groups + ['All']:
            # ---- get info by age -------------------
            if age != 'All':
                age_pop = self.P * self.age_distribution_list[self.age_groups.index(age)]
                age_sample = S_age == age
                age_noAccess_bmi = S_noAccess_bmi[age_sample]
                age_curr_bmi = S_curr_bmi[age_sample]
                age_curr_drug = S_curr_drug[age_sample]
                age_elig_bmi = S_elig_bmi[age_sample]
                age_elig_drug = S_elig_drug[age_sample]
            else:
                age_pop = self.P
                age_noAccess_bmi = S_noAccess_bmi
                age_curr_bmi = S_curr_bmi
                age_curr_drug = S_curr_drug
                age_elig_bmi = S_elig_bmi
                age_elig_drug = S_elig_drug

            # ----- get death info by age ----------------
            age_Dis_bmi_noAccess, age_De_noAccess_by_bmi, age_De_noAccess = (
                self.get_death_info_by_age(S_Dis_bmi_noAccess, age_noAccess_bmi, age_pop))
            age_Dis_bmi_curr, age_De_curr_by_bmi, age_De_curr = (
                self.get_death_info_by_age(S_Dis_bmi_noAccess, age_curr_bmi, age_pop))
            age_Dis_bmi_elig, age_De_elig_by_bmi, age_De_elig = (
                self.get_death_info_by_age(S_Dis_bmi_noAccess, age_elig_bmi, age_pop))

            # ------ get averted death by drugs ---------
            age_curr_ADe_drug = self.get_averted_death_by_drug(age_curr_drug, age_noAccess_bmi,
                                                               age_curr_bmi,
                                                               S_no_access_mu_all,
                                                               age_pop)

            age_elig_ADe_drug = self.get_averted_death_by_drug(age_elig_drug, age_noAccess_bmi,
                                                               age_elig_bmi,
                                                               S_no_access_mu_all,
                                                               age_pop)

            # ----- summary ------------------
            for i in range(self.num_bmi_cats + 1):
                if i < self.num_bmi_cats:
                    cat = self.bmi_categorization[i]
                    S_De_noAccess[age][cat] = age_De_noAccess_by_bmi[i]
                    S_De_curr[age][cat] = age_De_curr_by_bmi[i]
                    S_De_elig[age][cat] = age_De_elig_by_bmi[i]
                else:
                    S_De_noAccess[age]['All'] = age_De_noAccess
                    S_De_curr[age]['All'] = age_De_curr
                    S_De_elig[age]['All'] = age_De_elig

                    S_ADe_curr_dia[age], S_ADe_curr_obe[age] = age_curr_ADe_drug[1], age_curr_ADe_drug[2]
                    S_ADe_elig_dia[age], S_ADe_elig_obe[age] = (age_elig_ADe_drug[1] - age_curr_ADe_drug[1],
                                                                age_elig_ADe_drug[2] - age_curr_ADe_drug[2])

                    S_Dis_bmi_noAccess_by_age[age] = age_Dis_bmi_noAccess
                    S_Dis_bmi_curr_by_age[age] = age_Dis_bmi_curr
                    S_Dis_bmi_elig_by_age[age] = age_Dis_bmi_elig
        return (S_De_noAccess,
                S_De_curr, S_ADe_curr_dia, S_ADe_curr_obe,
                S_De_elig, S_ADe_elig_dia, S_ADe_elig_obe,
                S_Dis_bmi_noAccess_by_age, S_Dis_bmi_curr_by_age, S_Dis_bmi_elig_by_age)

    def get_death_summary(self, sample_path, samples):
        summary_path = sample_path + '/death_summary.pkl'
        if os.path.exists(summary_path):
            with open(summary_path, 'rb') as f:
                return pickle.load(f)

        Ss_age = samples['samples_age']
        Ss_noAccess_bmi = samples['samples_no_access_BMI']
        Ss_curr_bmi = samples['samples_current_BMI']
        Ss_curr_drug = samples['samples_current_drug']
        Ss_elig_bmi = samples['samples_eligible_BMI']
        Ss_elig_drug = samples['samples_eligible_drug']

        # ------- age distribution --------------
        Dis_age = []

        # ------- death info collection ----------
        De_noAccess = defaultdict(list)

        ADe_curr = defaultdict(list)
        ADe_curr_dia = defaultdict(list)
        ADe_curr_obe = defaultdict(list)

        ADe_elig = defaultdict(list)
        ADe_elig_dia = defaultdict(list)
        ADe_elig_obe = defaultdict(list)

        ADe_curr_and_elig_dia = defaultdict(list)
        ADe_curr_and_elig_obe = defaultdict(list)

        # -------- bmi distribution by age --------------
        Dis_bmi_noAccess_by_age = defaultdict(list)
        Dis_bmi_curr_by_age = defaultdict(list)
        Dis_bmi_elig_by_age = defaultdict(list)

        for i in range(len(Ss_age)):
            # get sample info
            S_age = Ss_age[i]
            S_noAccess_bmi = Ss_noAccess_bmi[i]
            S_curr_bmi = Ss_curr_bmi[i]
            S_curr_drug = Ss_curr_drug[i]
            S_elig_bmi = Ss_elig_bmi[i]
            S_elig_drug = Ss_elig_drug[i]

            # age distribution
            S_Dis_age = []
            for age_group in self.age_groups:
                S_Dis_age.append(sum(S_age == age_group) / len(S_age))
            Dis_age.append(S_Dis_age)

            # get death info
            (S_De_noAccess,
             S_De_curr, S_ADe_curr_dia, S_ADe_curr_obe,
             S_De_elig, S_ADe_elig_dia, S_ADe_elig_obe,
             S_Dis_bmi_noAccess_by_age, S_Dis_bmi_curr_by_age, S_Dis_bmi_elig_by_age) = (
                self.get_death_info(S_age, S_noAccess_bmi, S_curr_bmi, S_curr_drug, S_elig_bmi, S_elig_drug))

            # collect
            for age in S_De_noAccess:
                # ----- bmi distribution ------
                Dis_bmi_noAccess_by_age[age].append(S_Dis_bmi_noAccess_by_age[age])
                Dis_bmi_curr_by_age[age].append(S_Dis_bmi_curr_by_age[age])
                Dis_bmi_elig_by_age[age].append(S_Dis_bmi_elig_by_age[age])

                # ---- averted deaths ------
                ADe_curr_dia[age].append(S_ADe_curr_dia[age])
                ADe_curr_obe[age].append(S_ADe_curr_obe[age])

                ADe_elig_dia[age].append(S_ADe_elig_dia[age])
                ADe_elig_obe[age].append(S_ADe_elig_obe[age])

                ADe_curr_and_elig_dia[age].append(S_ADe_curr_dia[age] + S_ADe_elig_dia[age])
                ADe_curr_and_elig_obe[age].append(S_ADe_curr_obe[age] + S_ADe_elig_obe[age])

                for cat in S_De_noAccess[age]:
                    De_noAccess[age + ',' + cat].append(S_De_noAccess[age][cat])
                    ADe_curr[age + ',' + cat].append(S_De_noAccess[age][cat] - S_De_curr[age][cat])
                    ADe_elig[age + ',' + cat].append(S_De_curr[age][cat] - S_De_elig[age][cat])

        # ------- saving data -----------------
        summary_death = {
            # age distribution
            'age_distribution': Dis_age,

            # bmi distribution
            'Dis_bmi_noAccess_by_age': Dis_bmi_noAccess_by_age,
            'Dis_bmi_curr_by_age': Dis_bmi_curr_by_age,
            'Dis_bmi_elig_by_age': Dis_bmi_elig_by_age,

            # deaths
            'De_noAccess': De_noAccess,

            'ADe_curr': ADe_curr,
            'ADe_curr_dia': ADe_curr_dia,
            'ADe_curr_obe': ADe_curr_obe,

            'ADe_elig': ADe_elig,
            'ADe_elig_dia': ADe_elig_dia,
            'ADe_elig_obe': ADe_elig_obe,

            'ADe_curr_and_elig_dia': ADe_curr_and_elig_dia,
            'ADe_curr_and_elig_obe': ADe_curr_and_elig_obe,

        }
        with open(summary_path, 'wb') as f:
            pickle.dump(summary_death, f)
        return summary_death
