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
dy = 0.045;
for i=1:length(states)
    state = states(i);
    state = state{1,1};
    switch state
        case 'District of Columbia'
            text(6.088872053667636,2.970460276414367,0, state, FontSize=font_size-3);

        case 'Florida'
            text(7.643338099787916,3.082884234398095,0, state, FontSize=font_size-3);
        
        case 'Hawaii'
        text(6.705499862182752,2.673149550750514,0, state, FontSize=font_size-3);
        case 'Vermont'
            text(6.790939466699911,2.474140725170681,0, state, FontSize=font_size-3);
        case 'Massachusetts'
            text(7.871446412782623,2.540881167891565,0, state, FontSize=font_size-3);
        case 'Montana'
            text(8.190035878503236,2.846614675569804,0, state, FontSize=font_size-3);
        case 'New Hampshire'
            text(8.64487946298122,2.381487290615718,0, state, FontSize=font_size-3);
        case 'South Carolina'
            text(8.064573583362355,4.529960678722151,0, state, FontSize=font_size-3);
        case 'Georgia'
            text(9.038934826692865,4.026111384646409,0, state, FontSize=font_size-3);
        case  'New Jersey'
            text(7.64112934620815,3.232773105221378,0, state, FontSize=font_size-3);
        case 'Delaware'
            text(11.189227630113262,3.890195899263739,0, state, FontSize=font_size-3);
        case 'Rhode Island'
            text(7.822525446193488,3.330011239017002,0, state, FontSize=font_size-3);
        case 'Nevada'
            text(8.537926394756104,3.817178361193255,0, state, FontSize=font_size-3);
        case 'Washington'
            text(8.518016239377793,2.965409225764699,0, state, FontSize=font_size-3);
        case 'Louisiana'
            text(10.533554303536093,4.466640324574106,0, state, FontSize=font_size-3);
        case 'South Dakota'
            text(10.967867526082262,3.587947749528214,0, state, FontSize=font_size-3);
        case 'Mississippi'
            text(11.371758870670547,5.011946711515112,0, state, FontSize=font_size-3);
        case 'Pennsylvania'
            text(9.081038637830066,3.417764728585126,0, state, FontSize=font_size-3);
        case 'Missouri'
            text(9.872866552896514,3.643586297099151,0, state, FontSize=font_size-3);
        case 'Nebraska'
            text(11.063629801450109,3.301453648613165,0, state, FontSize=font_size-3);
        case 'Kansas'
            text(10.536258171592085,3.802620231985043,0, state, FontSize=font_size-3);
        case 'Wisconsin'
            text(11.375925609621998,2.834388308721176,0, state, FontSize=font_size-3);
        case 'North Carolina'
            text(8.874929070414804,3.906257030950412,0, state, FontSize=font_size-3);
        case 'Oregon'
            text(8.759214460883433,3.117949401753224,0, state, FontSize=font_size-3);
        case 'Maryland'
            text(9.530492782372145,3.18221735119998,0, state, FontSize=font_size-3);
        case 'Michigan'
            text(10.649602124080168,3.183305511480983,0, state, FontSize=font_size-3);
        case 'Arizona'
            text(8.893771283620623,3.530360237870244,0, state, FontSize=font_size-3);
        case  'Illinois'
            text(9.007797693012526,3.660062601768848,0, state, FontSize=font_size-3);
        case 'Virginia'
            text(9.644837647185517,3.803764101048497,0, state, FontSize=font_size-3);
        case 'Alabama'
            text(10.376681098129065,5.144890038809831,0, state, FontSize=font_size-3);
        otherwise
            if obesity_drug_death_per(i)<9.6
            text(obesity_drug_death_per(i)-9*dx, diabetes_drug_death_per(i)+dy, state, FontSize=font_size-3);
            else
                text(obesity_drug_death_per(i)+dx, diabetes_drug_death_per(i)+dy, state, FontSize=font_size-3);
            end
    end


end
xlim([6 13])
xlabel('Averted death per capita (obesity)')
ylabel('Averted death per capita (diabetes)')

set(gca,'FontSize',font_size)
saveas(gcf,'figs/fig3_temp.eps', 'epsc')
