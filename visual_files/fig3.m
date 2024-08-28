% fig 3
path = 'data/fig3_death_weight.xlsx';
death_table = readtable(path, 'Sheet', 'detailed', 'PreserveVariableNames',true);
states = death_table.Var1;
total_death_per = death_table.('total_death_per100000');
obesity_drug_death_per = death_table.('obesity_death_per100000');
diabetes_drug_death_per = death_table.('diabetes_death_per100000');
min_ = min(total_death_per);
max_ = max(total_death_per);
% --------- figure setting -----------------------
texts = char(65:128);
font_size = 14;
figure('Position',[688,351,581,474])

% --------- figure -----------------------
total_death_per_norm = (total_death_per-min(total_death_per))/(max(total_death_per)-min(total_death_per));

scatter(obesity_drug_death_per,diabetes_drug_death_per,...
    150, total_death_per,'filled')
cmp = colormap(flipud(cbrewer2("Spectral")));
cb = colorbar;
ylabel(cb,"Total averted death per capita", FontSize=font_size);
grid on
clim([min_ max_])

hold on
dx = 0.1;
dy = 0.05;
for i=1:length(states)
    state = states(i);
    state = state{1,1};
    switch state
        case 'Mississippi'
            text(10.87500079485211,4.864980851279745,0, state, FontSize=font_size-3);
        case 'Alabama'
            text(9.545401116467694,4.905944803076476,0, state, FontSize=font_size-3);
        case 'South Carolina'
            text(7.859550259582159,4.394243187866741,0, state, FontSize=font_size-3);
        case 'Louisiana'
            text(9.669711433065446,4.242471723431225,0, state, FontSize=font_size-3);
        case 'Georgia'
            text(8.717761597936864,3.901716123307664,0, state, FontSize=font_size-3);
        case 'Missouri'
            text(9.686066424391434,3.409734106528833,0, state, FontSize=font_size-3);
        case 'Ohio'
            text(10.233315534533135,3.987304333811341,0, state, FontSize=font_size-3);
        case 'Indiana'
            text(10.560593445116316,3.905208888591259,0, state, FontSize=font_size-3);
        case 'Minnesota'
            text(9.947426734677137,2.721720206285851,0, state, FontSize=font_size-3);
        case 'North Carolina'
            text(7.819776987199639,3.773382556944213,0, state, FontSize=font_size-3);
        case 'Illinois'
            text(8.63675086530072,3.543363423903065,0, state, FontSize=font_size-3);
        case 'Pennsylvania'
            text(8.723944018921717,3.297133996168238,0, state, FontSize=font_size-3);
        case 'Rhode Island'
            text(7.583410220429982,3.217766219929374,0, state, FontSize=font_size-3);
        case 'Florida'
            text(7.659960775414024,2.979723621696111,0, state, FontSize=font_size-3);
        case 'California'
            text(6.565347637694885,3.757056226787058,0, state, FontSize=font_size-3);
        case 'District of Columbia'
            text(6.039493996207853,2.907810113993359,0, state, FontSize=font_size-3);
        case 'Massachusetts'
            text(7.521204452726636,2.543083799422244,0, state, FontSize=font_size-3);
        case 'Alaska'
            text(9.371797458616017,2.535331033486349,0, state, FontSize=font_size-3);
        case 'Vermont'
            text(6.617081497281675,2.405915952502603,0, state, FontSize=font_size-3);
        case 'West Virginia'
            text(10.932334406657617,4.461293073476932,0, state, FontSize=font_size-3);
        case 'Connecticut'
            text(7.047904795564887,3.342495187271708,0, state, FontSize=font_size-3);
        case 'North Dakota'
            text(10.437753269862805,3.297353257385064,0, state, FontSize=font_size-3);
        otherwise
            if obesity_drug_death_per(i)<9.3
                text(obesity_drug_death_per(i)-length(state)/1.04*dx, diabetes_drug_death_per(i), state, FontSize=font_size-3);
            else
                text(obesity_drug_death_per(i)+dx, diabetes_drug_death_per(i), state, FontSize=font_size-3);
            end
    end


end
xlim([6 12])
xlabel('Averted death per capita (obesity)')
ylabel('Averted death per capita (diabetes)')

set(gca,'FontSize',font_size)
saveas(gcf,'figs/fig3_temp.eps', 'epsc')
