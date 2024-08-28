% fig2
path = 'data/fig2.xlsx';

% ------ get data ----------------
BMI_categorization = {'Underweight', 'Normal', 'Overweight-I', 'Overweight-II',...
    'Obesity-I', 'Obesity-II', 'Obesity-III'};
BMI_low =  [0,   18.5, 25.0,27.5,30.0,35.0,40.0];
BMI_high = [18.5,25.0, 27.5,30.0,35.0,40.0,100.0];

value_distribution = readtable(path, 'Sheet', 'mean', 'PreserveVariableNames',true);
value_dx = value_distribution.x;

no_access_distribution = value_distribution.mean_no_access;
current_distribution = value_distribution.mean_current;
eligible_distribution = value_distribution.mean_eligible;
eligible_distribution_hypo = value_distribution.mean_eligible_hypo;

% --------- figure setting -----------------------
marker_size = 6;
line_width = 1.5;
font_size = 14;
face_alpha = 0.4;
no_access_color = '#dfc27d';
current_color = '#a6611a';
eligible_color ='#5ab4ac';
eligible_color_hypo = '#01665e';
BMI_color = {'#ffffff', '#d9d9d9', '#969696', '#737373', '#525252', '#252525', '#000000'};
figure('Position',[138,572,769,344])

% ------- draw value distribution --------------
figs = 1;
subplot(1,1,figs,'Units','normalized')
x_ticks = [15, 18.5, 25, 27.5, 30, 35, 40, 45, 50, 55];
% ------ bmi categorization ------
bmi_min = 15; %min(value_dx)
bmi_max = 55; %max(value_dx)
probability_lim = 0.1;
for bmi_index=1:length(BMI_categorization)
    low_bmi_data = max(bmi_min, BMI_low(bmi_index));
    high_bmi_data = min(bmi_max, BMI_high(bmi_index));
    bmi_cat_position = low_bmi_data+0.3;
    xregion(low_bmi_data,high_bmi_data,'FaceColor', hex2rgb(BMI_color(bmi_index)),'FaceAlpha', 0.2);
    hold on
    text(bmi_cat_position, probability_lim-0.06, BMI_categorization(bmi_index), 'FontSize',font_size-2,'FontWeight','bold',...
        'Rotation',90, 'VerticalAlignment','top', 'HorizontalAlignment','left','Color', hex2rgb('#737373'))
end
% ------ bmi value distribution -------
h_all = [];

% no access
fill_curve(value_dx,zeros(size(value_dx)),no_access_distribution, hex2rgb(no_access_color),face_alpha,line_width, 'none',hex2rgb(no_access_color));
hold on
plot(value_dx,no_access_distribution,'-.','LineWidth',line_width,'Color',hex2rgb('#404040'));
hold on
h = fill_curve(value_dx(1),zeros(1),zeros(1), hex2rgb(no_access_color),face_alpha,line_width, '-.',hex2rgb('#404040'));
h_all = [h_all, h];
hold on

% current uptake
fill_curve(value_dx,zeros(size(value_dx)),current_distribution, hex2rgb(current_color), face_alpha,line_width, 'none',hex2rgb(current_color));
hold on
plot(value_dx,current_distribution,'-','LineWidth',line_width-1,'Color',hex2rgb(current_color))
hold on
h = fill_curve(value_dx(1),zeros(1),zeros(1), hex2rgb(current_color), face_alpha,line_width-1, '-',hex2rgb(current_color));
h_all = [h_all, h];
hold on

% eligible
fill_curve(value_dx,zeros(size(value_dx)),eligible_distribution, hex2rgb(eligible_color), face_alpha,line_width, 'none', hex2rgb(eligible_color));
hold on
plot(value_dx,eligible_distribution,'-','LineWidth',line_width,'Color',hex2rgb(eligible_color))
hold on
h = fill_curve(value_dx(1),zeros(1),zeros(1), hex2rgb(eligible_color), face_alpha,line_width, '-',hex2rgb(eligible_color));
h_all = [h_all, h];

% eligible (optimistic)
fill_curve(value_dx,zeros(size(value_dx)),eligible_distribution_hypo, hex2rgb(eligible_color_hypo), face_alpha,line_width, 'none', hex2rgb(eligible_color_hypo));
hold on
plot(value_dx,eligible_distribution_hypo,'-','LineWidth',line_width-1,'Color',hex2rgb(eligible_color_hypo))
hold on
h = fill_curve(value_dx(1),zeros(1),zeros(1), hex2rgb(eligible_color_hypo), face_alpha,line_width-1, '-',hex2rgb(eligible_color_hypo));
h_all = [h_all, h];


title('Adults (aged 18 and over) BMI distribution','FontWeight','normal')
xticks(x_ticks)
xlabel('BMI [kg/m^2]')
ylabel('Probability Density')
xlim([bmi_min, bmi_max])
ylim([0, probability_lim])
legend(h_all, 'No access', 'Current uptake', 'Expanded access',  'Expanded access (optimistic)')
set(gca,'FontSize',font_size)
saveas(gcf,'figs/fig2.png')

set(gcf, 'PaperPosition', [0 0 12*769/344 12]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [12*769/344 12]); %Set the paper to have width 5 and height 5.
saveas(gcf,'figs/fig2.pdf')

