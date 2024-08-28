import squarify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch
import utils
from collections import defaultdict

# ---- input data -----
path = 'data/fig1.xlsx'
# fig A
bmi_distribution_table = pd.read_excel(path, sheet_name='bmi distribution').to_dict()

# fig B
uptake_info_table = pd.read_excel(path, sheet_name='uptake_scenario_info', index_col='Unnamed: 0')
current_diabetes = uptake_info_table.loc['current_diabetes']
no_diabetes = uptake_info_table.loc['no_diabetes']
current_obesity = uptake_info_table.loc['current_obesity']
no_obesity = uptake_info_table.loc['no_obesity']
merged_cats = uptake_info_table.columns

fig1B_SM = defaultdict(dict)
elig_obesity = round(uptake_info_table.sum() * 100, 1)
fig1B_SM['Eligible for obesity drugs'] = {'Overweight (%)': str(elig_obesity['Overweight']),
                                          'Obesity (%)': str(elig_obesity['Obesity'])}
elig_diabetes = round((current_diabetes + no_diabetes) * 100, 1)
fig1B_SM['Eligible for diabetes drugs'] = {'Overweight (%)': str(elig_diabetes['Overweight']),
                                           'Obesity (%)': str(elig_diabetes['Obesity'])}
fig1B_SM['Currently using obesity drugs'] = {'Overweight (%)': str(round(current_obesity['Overweight'] * 100, 1)),
                                             'Obesity (%)': str(round(current_obesity['Obesity'] * 100, 1))}
fig1B_SM['Currently using diabetes drugs'] = {'Overweight (%)': str(round(current_diabetes['Overweight'] * 100, 1)),
                                              'Obesity (%)': str(round(current_diabetes['Obesity'] * 100, 1))}
fig1B_SM = pd.DataFrame(fig1B_SM).transpose()
fig1B_SM.to_excel('data/SM_fig1B.xlsx')
# fig C
eligibility_among_insurance = {}
eligibility_among_insurance_table = pd.read_excel(path, sheet_name='eligibility among insurance').to_dict()
for insurance in eligibility_among_insurance_table:
    eligibility_among_insurance[insurance] = eligibility_among_insurance_table[insurance][0]

bmi_cats = ['Underweight', 'Normal',
            'Overweight-grade1', 'Overweight-grade2',
            'Obesity-grade1', 'Obesity-grade2', 'Obesity-grade3']

ordered_bmi_cats = ['Underweight', 'Normal',
                    'Overweight-I', 'Overweight-II',
                    'Obesity-I', 'Obesity-II', 'Obesity-III']
bmi_distribution_values = []
bmi_distribution_labels = []
for i in range(len(ordered_bmi_cats)):
    bmi_distribution_values.append(bmi_distribution_table[bmi_cats[i]][0])
    bmi_distribution_labels.append(ordered_bmi_cats[i] + '\n' + utils.fmt(bmi_distribution_values[i]))
# ----- figs settings -------
cats_face_color = {'Underweight': 'fee08b',
                   'Normal': 'abdda4',
                   'Overweight-I': 'fdae61',
                   'Overweight-II': 'fdae61',
                   'Obesity-I': 'd7191c',
                   'Obesity-II': 'd7191c',
                   'Obesity-III': 'd7191c'}
dash_color = '#404040'
dash_width = 4
plt.rc('font', family='Helvetica')
font = {
    'weight': 'bold',
    'size': 38,
}

# ----- fig1 tree map -------
alpha = 0.8
fig_all = plt.figure(figsize=(28, 18), layout=None)
ax1 = plt.subplot(2, 2, 1)
fig1_x, fig1_y, fig1_w, fig1_h = 0.02, 0.4, 0.48, 0.58
ax1.set_position([fig1_x, fig1_y, fig1_w, fig1_h])

# tree map
s = squarify.plot(bmi_distribution_values, label=bmi_distribution_labels, text_kwargs={'fontsize': font['size'] + 2})
plt.axis('off')

# add face color
for rect, cat in zip(s.patches, ordered_bmi_cats, ):
    if cat == 'Underweight':
        rect.set_facecolor(utils.c_without_alpha(cats_face_color[cat], 1))
    else:
        rect.set_facecolor(utils.c_without_alpha(cats_face_color[cat], alpha))
    if cat == 'Overweight-I':
        ov1_x, ov1_y = rect.xy
    if cat == 'Overweight-II':
        ov2_x, ov2_y = rect.xy
        ov2_h = rect._height
        ov2_w = rect._width
    if cat == 'Obesity-II':
        ob2_x, ob2_y = rect.xy
    if cat == 'Obesity-I':
        ob1_h = rect._height
    if cat == 'Obesity-III':
        ob3_x, ob3_y = rect.xy
        ob3_w = rect._width

# dash between overweight 1 and 2
dash_between_overweight_x = [ov2_x, ov2_x]
dash_between_overweight_y = [ov2_y, ov2_y + ov2_h]
plt.plot(dash_between_overweight_x, dash_between_overweight_y, linestyle='dashed', color=dash_color,
         linewidth=dash_width)

# dash between obesity 1 and obesity 2/3
dash_between_obesity123_x = [ob2_x, ob2_x]
dash_between_obesity123_y = [ob2_y, ob2_y + ob1_h]
plt.plot(dash_between_obesity123_x, dash_between_obesity123_y, linestyle='dashed', color=dash_color,
         linewidth=dash_width)

# dash between obesity 2 and 3
dash_between_obesity23_x = [ob3_x, ob3_x + ob3_w]
dash_between_obesity23_y = [ob3_y, ob3_y]
plt.plot(dash_between_obesity23_x, dash_between_obesity23_y, linestyle='dashed', color=dash_color, linewidth=dash_width)

# ----- fig2 eligibility among overweight and obesity -------
ax2 = plt.subplot(2, 2, 2)
mpl.rcParams['hatch.linewidth'] = 3
height = 0.4
fig2_x = 0.1
fig2_y = 0.08
fig2_w = fig1_x + fig1_w - fig2_x
fig2_h = 0.32 - fig2_y
ax2.set_position([fig2_x, fig2_y, fig2_w, fig2_h])

color_eligible_diabetes = utils.c_without_alpha('2b83ba', 1)
color_eligible_obesity = utils.c_without_alpha('2b83ba', 0.4)

plt.barh(merged_cats, current_diabetes, height, color=color_eligible_diabetes, hatch='//')
plt.barh(merged_cats, no_diabetes, height, left=current_diabetes, color=color_eligible_diabetes)
plt.barh(merged_cats, current_obesity, height, left=current_diabetes + no_diabetes, color=color_eligible_obesity,
         hatch='//', )
plt.barh(merged_cats, no_obesity, height, left=current_diabetes + no_diabetes + current_obesity,
         color=color_eligible_obesity, )
ax2.invert_yaxis()  # labels read top-to-bottom
plt.xlabel('Percentage (%)', font=font)
ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], labels=[0, 20, 40, 60, 80, 100], font=font)
ax2.set_yticks(np.arange(len(merged_cats)), labels=merged_cats, font=font)

handles = []
handles.append(mpatches.Patch(facecolor=color_eligible_diabetes, label='Eligible for diabetes/obesity drugs'))
handles.append(mpatches.Patch(facecolor=color_eligible_obesity, label='Eligible only for obesity drugs'))
handles.append(mpatches.Patch(edgecolor=dash_color, facecolor='white', hatch="//", label='Current uptake'))
plt.legend(handles=handles, frameon=False, fontsize=font['size'], loc='upper right', bbox_to_anchor=(1.03, 1))

# zoom in lines
con = ConnectionPatch(xyA=(ov1_x, ov1_y), xyB=(0, 1), coordsA="data", coordsB=ax2.get_xaxis_transform(),
                      axesA=ax1, axesB=ax1, linestyle='--', color=dash_color, linewidth=dash_width - 2)
ax2.add_artist(con)
con = ConnectionPatch(xyA=(ov2_x + ov2_w, ov2_y), xyB=(1.05, 1), coordsA="data", coordsB=ax2.get_xaxis_transform(),
                      axesA=ax1, axesB=ax1, linestyle='--', color=dash_color, linewidth=dash_width - 2)
ax2.add_artist(con)

# ----- eligibility across insurance ----------
ax = plt.subplot(2, 2, 3)
fig3_y = 0.75
ax.set_position([0.02 + 0.645, fig3_y, 0.32, 0.22])

# bar plot
insurance_types = list(eligibility_among_insurance.keys())
y_pos = np.arange(len(insurance_types))
share_eli = np.array(list(eligibility_among_insurance.values())) * 100
plt.xlim([20, 60])
ax.barh(y_pos, share_eli, 0.6, align='center', color='#2b83ba')
font_bar = {
    'weight': 'bold',
    'size': font['size'] - 3,
}
for i, v in enumerate(share_eli):
    ax.text(v + 0.3, i, str(round(v)), color='black', verticalalignment='center', fontdict=font_bar)
ax.set_yticks(y_pos, labels=insurance_types, font=font)
ax.set_xticks([20, 30, 40, 50, 60], labels=[20, 30, 40, 50, 60], font=font)
ax.invert_yaxis()  # labels read top-to-bottom
plt.xlabel('Percentage (%)', font=font)
plt.ylabel('Insurance category', font=font)

# ----- eligibility across states ----------
ax = plt.subplot(2, 2, 4)
ax.set_position([fig2_x + fig2_w + 0.05, fig2_y, 0.32, fig3_y - fig2_y])
plt.text(0.5, -0.06, 'Percentage (%)', font=font)
plt.axis('off')
plt.show()

fig_all.savefig('figs/fig1_temp.eps')
