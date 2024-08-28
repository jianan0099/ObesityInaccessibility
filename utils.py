import numpy as np
from collections import defaultdict
from scipy.stats import norm
import scipy
from scipy.stats import beta
from scipy.optimize import minimize
import pandas as pd


def save_dfs(file_path, dfs, sheet_names, indexes):
    dfs[0].to_excel(file_path, sheet_name=sheet_names[0], index=indexes[0])
    for i in range(1, len(dfs)):
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            dfs[i].to_excel(writer, sheet_name=sheet_names[i], index=indexes[i])


def percentile_range_linear(x_min_, x_max_, p_min_, p_max_):
    slope_ = (p_max_ - p_min_) / (x_max_ - x_min_)
    intercept_ = p_max_ - slope_ * x_max_
    return slope_, intercept_


def linear_data_interpolation(available_percentiles_, available_data_, target_percentiles_):
    target_data = []
    for target_p in target_percentiles_:
        if target_p < available_percentiles_[0]:
            slope_right, intercept_right = percentile_range_linear(available_data_[0], available_data_[1],
                                                                   available_percentiles_[0], available_percentiles_[1])
            target_data.append((target_p - intercept_right) / slope_right)
        elif target_p > available_percentiles_[-1]:
            slope_left, intercept_left = percentile_range_linear(available_data_[-2], available_data_[-1],
                                                                 available_percentiles_[-2], available_percentiles_[-1])
            target_data.append((target_p - intercept_left) / slope_left)
        else:
            target_data.append(np.interp(target_p, available_percentiles_, available_data_))
    return target_data


def get_ecdf(full_percentiles_, full_percentile_data_, x_):
    return np.interp(x_, xp=full_percentile_data_, fp=full_percentiles_ * 0.01, left=0, right=1)


def get_pdf_from_empirical_cdf_general(full_data_, full_percentiles_, data_):
    pdf = np.zeros_like(data_) * 1.0
    non_zero_pdf_index = np.arange(len(data_))[(data_ >= full_data_[0]) & (data_ <= full_data_[-1])]
    for index in non_zero_pdf_index:
        for i in range(len(full_percentiles_) - 1):
            if full_data_[i] <= data_[index] <= full_data_[i + 1]:
                slope_, _ = percentile_range_linear(full_data_[i], full_data_[i + 1],
                                                    full_percentiles_[i] * 0.01, full_percentiles_[i + 1] * 0.01)
                pdf[index] = slope_
                break
    return pdf


def generate_sex_age_group_sample(percentile_info_W_, percentile_info_H_, percentile_info_BMI_, n):
    full_percentiles = np.array([0, 5, 10, 15, 25, 50, 75, 85, 90, 95, 100])
    samples = defaultdict(list)
    remaining_n = n
    while remaining_n > 0:
        w = np.interp(np.random.uniform(0, 1, remaining_n), full_percentiles * 0.01, percentile_info_W_)
        h = np.interp(np.random.uniform(0, 1, remaining_n), full_percentiles * 0.01, percentile_info_H_)
        bmi = w / ((h / 100) ** 2)
        bmi_eligible_index = np.arange(remaining_n)[
            (percentile_info_BMI_[0] <= bmi) & (bmi <= percentile_info_BMI_[-1])]
        lik_eligible_index = get_pdf_from_empirical_cdf_general(percentile_info_BMI_,
                                                                full_percentiles,
                                                                bmi[bmi_eligible_index])
        final_eligible_index = bmi_eligible_index[lik_eligible_index < np.random.random(len(lik_eligible_index))]
        samples['weight'] += list(w[final_eligible_index])
        samples['height'] += list(h[final_eligible_index])
        samples['BMI'] += list(bmi[final_eligible_index])
        remaining_n = n - len(samples['weight'])
    return samples


def full_percentile_info(row, full_percentiles):
    percentile_columns = [str(p) + 'th' for p in full_percentiles]
    return np.array(row[percentile_columns].values, dtype=np.float64)


def mean_confidence_interval(data, confidence):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def b_cal(a, weight_loss_mean_norm):
    return (1 - weight_loss_mean_norm) / weight_loss_mean_norm * a


def objective(a, weight_loss_mean_norm, weight_loss_norm, cdfs_norm):
    b = b_cal(a, weight_loss_mean_norm)
    if a < 0:
        value = np.inf
    elif b < 0:
        value = np.inf
    else:
        theoretical_cdfs = beta.cdf(weight_loss_norm, a=a, b=b)
        value = np.sum((theoretical_cdfs - cdfs_norm) ** 2)
    return value


def eff_beta_fit(weight_loss_mean, weight_loss: list, percentages: list, min_weight_loss, max_weight_loss):
    weight_loss_norm = np.array([min_weight_loss] + weight_loss + [max_weight_loss])
    weight_loss_norm = (weight_loss_norm - min_weight_loss) / (max_weight_loss - min_weight_loss)
    cdfs_norm = 1 - np.array([1.0] + percentages + [0.0])
    weight_loss_mean_norm = (weight_loss_mean - min_weight_loss) / (max_weight_loss - min_weight_loss)
    result = minimize(objective, x0=np.array([1.0]), method='BFGS', args=(weight_loss_mean_norm,
                                                                          weight_loss_norm,
                                                                          cdfs_norm))
    a_hat = result.x
    b_hat = b_cal(a_hat, weight_loss_mean_norm)
    return a_hat[0], b_hat[0]


def save_percentage_str(percentage, round_num):
    return '{:.2f}'.format(round(percentage*100, round_num))


def c_without_alpha(hex_, alpha_):
    rgb_value = tuple(int(hex_[i_:i_ + 2], 16) for i_ in (0, 2, 4))
    new_rgb_value = []
    for i_ in range(len(rgb_value)):
        new_rgb_value.append(rgb_value[i_] * alpha_ + 255 * (1 - alpha_))
    return tuple(np.array(new_rgb_value) / 255)


def fmt(x):
    return '{:.1f}%'.format(x * 100)


def number_with_comma(number):
    modified_number = ''
    number = str(number)
    flag = 0
    for i in range(len(number)-1, -1, -1):
        flag += 1
        if flag == 3 and i > 0:
            flag = 0
            modified_number = ',' + number[i] + modified_number
        else:
            modified_number = number[i] + modified_number
    return modified_number

