function f_cdf_compare(ecdf_p,ecdf_x_T, group_order, measure, T_actual_mean,...
    measure_label, xticks_value,path)
T_sample_mean = readtable(path, 'Sheet', strcat(measure,'_sample_mean'),'PreserveVariableNames',true);

% --------- figure setting -----------------------
texts = char(65:128);
marker_size = 6;
line_width = 2;
font_size = 14;
actual_color = [44/255,123/255,182/255];
sample_color = [215/255,25/255,28/255];
figure('Position',[473,282,824,639])

% ---------- subplot -----------------------------
figs = 1;
for row=1:4
    if row<=2
        col_num = 5;
    else
        col_num = 4;
    end
    for col=1:col_num
        % selected group
        selected_group = char(group_order(figs));
        selected_group_info = split(selected_group, ', ');
        selected_age = string(selected_group_info(2));
        % ecdf
        ecdf_x = ecdf_x_T.(selected_group);
        % actual mean
        actual_mean = T_actual_mean(string(T_actual_mean.Var1)==selected_group,measure);
        actual_mean = actual_mean{1,1};

        % percentiles of samples
        sample_x_mean = table2array(readtable(path, 'Sheet', strcat(measure,'_',selected_group(1),'_', selected_age),'PreserveVariableNames',true));
        sample_x_mean = sample_x_mean(2,:);
        % sample mean
        sample_mean_mean = T_sample_mean.(selected_group);
 

        % draw fig
        subplot(4,5,(row-1)*5+col, 'Position', [0.07+(col-1)*0.185, 0.78-(row-1)*0.23, 0.15,0.16],'Units','normalized')
        h_all = [];
        
        h = plot(ecdf_x, ecdf_p*100, '-+','LineWidth', line_width, ...
            'color', actual_color , 'MarkerSize',marker_size);
        h_all = [h_all, h];
        hold on

        h = xline(actual_mean,'-','LineWidth', line_width, ...
            'color', actual_color );
        h_all = [h_all, h];
        hold on
        
        h = plot(sample_x_mean, ecdf_p*100, '--','LineWidth', line_width, ...
            'color', sample_color);
        h_all = [h_all, h];
        hold on

        h = xline(sample_mean_mean,':', 'LineWidth', line_width, ...
            'color', sample_color);
        h_all = [h_all, h];
        
        
        title(selected_group,'FontWeight','normal')

        if col>1
            set(gca,'Yticklabel',[]);
        end

        if row==4 || (row==2 && col==5)
            xlim([xticks_value(1), xticks_value(end)]);
            xticks(xticks_value);
            xlabel(measure_label);
        else
            xlim([xticks_value(1), xticks_value(end)]);
            xticks(xticks_value);
            set(gca,'Xticklabel',[]);
        end

        if row==3 && col==1
            ylabel('Probability (%)', 'Units', 'Normalized','Position',[-0.276612903225806,1.343137731739119,0]);
        end

        text(-0.09, 1.25, texts(figs), 'Units', 'Normalized','FontSize',font_size,'FontWeight','bold');
        set(gca,'FontSize',font_size)

        figs=figs+1;
    end
end
legend(h_all,'Empirical CDF', 'Empirical mean', 'Sample CDF','Sample mean',...
    'Units', 'Normalized','Position', [0.816913972575315,0.215962441314554,0.135755930337305,0.156580746125499])
end