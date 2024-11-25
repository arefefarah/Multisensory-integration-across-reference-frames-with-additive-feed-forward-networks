
clear;
load('net_refframe_and_multisensory_5lay_50000_64')

%% analyze the output of  the network
load('network_out.mat')
%plot position
[m, b, r] = postreg(desired_position,network_pos_output);
xlabel("Desired position (deg)")
ylabel("Decoded position (deg)")
xlim([-100 100])
ylim([-100 100])
legend('off');
title('');
text(10, -40, 'R^2= 0.89', 'FontSize', 12, 'Color', 'black');
text(10, -55, 'Slope = 0.99', 'FontSize', 12, 'Color', 'black');
saveas(gcf, 'position.png');



error_position = abs(desired_position-network_pos_output);
fprintf("mean")
mean(error_position)
fprintf("sigma")
std(error_position)
binEdges = 0:2:40;
figure
h = histogram(error_position,binEdges); xlabel("Position error (deg)");ylabel("# Observations");
h.FaceColor = [0.2,0.55,0.2]; % RGB triplet for teal color 
h.EdgeColor = 'w'; % White edges
h.FaceAlpha = 0.6; % Transparency level
% saveas(gcf, 'error_position.png');



%plot variance
figure
[m, b, r] = postreg(desired_variance,network_var_output);
xlabel("Desired variance (deg^2)")
ylabel("Decoded variance (deg^2)")
legend('off');
title('');
text(14, 7, 'R^2= 0.99', 'FontSize', 12, 'Color', 'black');
text(14, 5, 'Slope = 1', 'FontSize', 12, 'Color', 'black');
% saveas(gcf, 'variance.png');

error_var = abs(desired_variance-network_var_output);
fprintf("mean")
mean(error_var)
fprintf("sigma")
std(error_var)
binEdges = -0:0.1:3.5;
figure
h = histogram(error_var,binEdges); xlabel("Variance error (deg^2)");ylabel("# Observations");
h.FaceColor = [0.2,0.55,0.2]; % RGB triplet for teal color 
h.EdgeColor = 'w'; % White edges
h.FaceAlpha = 0.6; % Transparency level
% saveas(gcf, 'error_variance.png');