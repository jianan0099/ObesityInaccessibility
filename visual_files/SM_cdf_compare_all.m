% supple
path = strcat('data/SM_cdf_compare.xlsx');
T_actual_mean = readtable(path, 'Sheet', 'actual_mean','PreserveVariableNames',true);
measures = {'weight', 'height', 'BMI'};
group_order = {'Male, 18 years', 'Male, 19 years', 'Male, 20–29', 'Male, 30–39', 'Male, 40–49',...
         'Female, 18 years', 'Female, 19 years', 'Female, 20–29', 'Female, 30–39', 'Female, 40–49',...
         'Male, 50–59', 'Male, 60–69', 'Male, 70–79', 'Male, 80 and over', ...
         'Female, 50–59', 'Female, 60–69', 'Female, 70–79', 'Female, 80 and over', ...
    };
measure_xticks = [40, 80, 120, 160;
                  140, 160, 180, 200;
                  15, 30, 45, 60];
measure_labels = {'Weight [kg]', 'Height [cm]', 'BMI [kg/m^2]'};

for measure_index=1:length(measures)
    
    measure = string(measures(measure_index));
    T_actual = readtable(path, 'Sheet', strcat(measure, '_actual'),'PreserveVariableNames',true);
    ecdf_p = T_actual.('percentiles');
    f_cdf_compare(ecdf_p,T_actual, group_order, measure,T_actual_mean,...
        measure_labels{measure_index},...
        measure_xticks(measure_index,1:end), path)
    saveas(gcf,strcat('figs/sample_',measure,'.eps'), 'epsc')
end
