function h=fill_curve(x,y_low,y_high, color,face_alpha,line_width, line_style, edge_color)
x_fill = [x; flip(x, 1)];
inbetween = [y_low; flip(y_high,1)];
h=fill(x_fill, inbetween, color,'FaceAlpha', face_alpha, 'LineStyle',line_style, 'EdgeColor',edge_color,'LineWidth',line_width);
end
