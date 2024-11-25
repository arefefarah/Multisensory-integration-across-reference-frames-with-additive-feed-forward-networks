load('net_refframe_and_multisensory_5lay_50000_64')

temp1 = zeros(1,11)+10; % retinal hand positions     
temp2 = zeros(1,11)-10;  % random set of associated proprioceptive hand position  
temp3 = 20; % reference frame transformation angle
temp4 = (2.5:0.5:7.5).^2; 
n1 = length(temp1); n2 = length(temp2); n3 = length(temp3);
ff = 50; xp = 2;
%% bimodal Response

Ainput = [repmat(temp1',n2,1) reshape(repmat(temp2',1,n1)',n1*n2,1) repmat(temp3,n2*n1,1)]';
AVarVis = repmat(temp4,1,n2)+0.00001; % visual variance
AVarPro = reshape(repmat(temp4,n1,1),1,n2*n2)+0.00001; % proprioceptive variance
AVarEye = repmat(3.5.^2,1,n1*n2); % eye position variance

AtunVis = exp(-(repmat(x',1,n1*n2)-repmat(Ainput(1,:),Ni,1)).^2./10.^2./2);
AampVis = ff./AVarVis;
AactVis = repmat(AampVis, Ni, 1).*AtunVis;
Pdana{1,1} = AactVis./ff; % activations of 1-D retinal map

AtunPro = poslin(repmat(offset',1,n1*n2) + repmat(slope',1,n1*n2).*repmat(Ainput(2,:),Ne,1)./pMax);
AampPro = ff./AVarPro;
AactPro = repmat(AampPro, Ne, 1).*AtunPro;
Pdana{2,1} = AactPro./ff; % proprioceptive coding
if xp == 2,
    AtunEye = poslin(repmat(offset2',1,n1*n2) + repmat(slope2',1,n1*n2).*repmat(Ainput(3,:),Ne,1)./eMax);
    AampEye = ff./AVarEye;
    AactEye = repmat(AampEye, Ne, 1).*AtunEye;
    Pdana{3,1} = AactEye/ff; % proprioceptive coding
end
% simulate net
net.outputConnect = [1 1 1];
ZA_bi = sim(net,Pdana); % simulate trained network

lay = 2;
[XX,YY] = meshgrid((0.5:0.5:5.5).^2, (0.5:0.5:5.5).^2);
RR_bi=zeros(64,11*11);
for q=0:3
    for j =1+(q*16):(q+1)*16
        RR_bi(j,:) = ZA_bi{lay}(j,:);

    end
end

AVarTot = ((AVarVis+AVarEye).*AVarPro)./(AVarVis+AVarEye+AVarPro);
Aoutput = AVarTot.*((Ainput(1,:)+Ainput(2,:))./(AVarVis+AVarEye) + Ainput(3,:)./AVarPro); % spatial target position

 %% unimodal resp for vis only
Ainput = [repmat(temp1',n2,1) reshape(repmat(temp2',1,n1)',n1*n2,1) repmat(temp3,n2*n1,1)]';
AVarVis = repmat(temp4,1,n2)+0.00001; % visual variance
AVarPro = repmat(500.^2,1,n1*n2); % proprioceptive variance
AVarEye = repmat(3.5.^2,1,n1*n2); % eye position variance

AtunVis = exp(-(repmat(x',1,n1*n2)-repmat(Ainput(1,:),Ni,1)).^2./10.^2./2);
AampVis = ff./AVarVis;
AactVis = repmat(AampVis, Ni, 1).*AtunVis;
Pdana_vis{1,1} = AactVis./ff; % activations of 1-D retinal map

AtunPro = poslin(repmat(offset',1,n1*n2) + repmat(slope',1,n1*n2).*repmat(Ainput(2,:),Ne,1)./pMax);
AampPro = ff./AVarPro;
AactPro = repmat(AampPro, Ne, 1).*AtunPro;
Pdana_vis{2,1} = AactPro./ff; % proprioceptive coding
if xp == 2,
    AtunEye = poslin(repmat(offset2',1,n1*n2) + repmat(slope2',1,n1*n2).*repmat(Ainput(3,:),Ne,1)./eMax);
    AampEye = ff./AVarEye;
    AactEye = repmat(AampEye, Ne, 1).*AtunEye;
    Pdana_vis{3,1} = AactEye/ff; % proprioceptive coding
end

net.outputConnect = [1 1 1];
ZA_uni_vis = sim(net,Pdana_vis); % simulate trained network
% here I fixed varpro = 5000^2 => constant visual reliability (no proprioceptive info available)
lay = 2;
RR_uni_vis=zeros(64,11*11);
for q=0:3
    for j =1+(q*16):(q+1)*16
        RR_uni_vis(j,:) = ZA_uni_vis{lay}(j,:);
    end
end
%% unimodal response for prop only
Ainput = [repmat(temp1',n2,1) reshape(repmat(temp2',1,n1)',n1*n2,1) repmat(temp3,n2*n1,1)]';
AVarVis = repmat(5000.^2,1,n1*n2); %here I fixed varvis = 5000^2 => constant prop reliability (no visual info available)
AVarPro = reshape(repmat(temp4,n1,1),1,n2*n2)+0.00001;% proprioceptive variance
AVarEye = repmat(3.5.^2,1,n1*n2); % eye position variance

AtunVis = exp(-(repmat(x',1,n1*n2)-repmat(Ainput(1,:),Ni,1)).^2./10.^2./2);
AampVis = ff./AVarVis;
AactVis = repmat(AampVis, Ni, 1).*AtunVis;
Pdana_prop{1,1} = AactVis./ff; % activations of 1-D retinal map

AtunPro = poslin(repmat(offset',1,n1*n2) + repmat(slope',1,n1*n2).*repmat(Ainput(2,:),Ne,1)./pMax);
AampPro = ff./AVarPro;
AactPro = repmat(AampPro, Ne, 1).*AtunPro;
Pdana_prop{2,1} = AactPro./ff; % proprioceptive coding
if xp == 2,
    AtunEye = poslin(repmat(offset2',1,n1*n2) + repmat(slope2',1,n1*n2).*repmat(Ainput(3,:),Ne,1)./eMax);
    AampEye = ff./AVarEye;
    AactEye = repmat(AampEye, Ne, 1).*AtunEye;
    Pdana_prop{3,1} = AactEye/ff; % proprioceptive coding
end

net.outputConnect = [1 1 1];
ZA_uni_prop = sim(net,Pdana_prop); % simulate trained network
% here I fixed varpro = 5000^2 => constant visual reliability (no proprioceptive info available)
lay = 2;
RR_uni_prop=zeros(64,11*11);
for q=0:3
    for j =1+(q*16):(q+1)*16
        RR_uni_prop(j,:) = ZA_uni_prop{lay}(j,:);
    end
end

%% plot new additivity 
reliability = flip(1./temp4); %%  convert variance to reliability (1./var)
r = linspace(0,10,11);
[XX,YY] = meshgrid(r, r); % 

for q=0:3
    figure
    for j =1+(q*16):(q+1)*16
        subplot(4,4,j-16*q);
        RR = reshape(RR_bi(j,:),size(XX));
        RR_vis = reshape(RR_uni_prop(j,:),size(XX));  %RR(1,:);
        RR_prop = reshape(RR_uni_vis(j,:),size(XX));  %RR(:,1);
        Ri = RR./(RR_vis+RR_prop);
        surf(XX,YY,flip(flip(Ri,1),2)); % or rot90(Ri,2)  %% reorder additivity based on reliability not variance
        title(j)
%         axis([0.25 30.25 0.25 30.25]); %axis 'auto z'
        view([0 0 10]); shading interp
        if (j == 4), xlabel('Proprioceptive reliability (a.u.)'); end
        if (j == 4), ylabel('Visual reliability (a.u.) '); end
        if (j == 2), title(['\bf{Additivity index, layer ' num2str(lay) '}']); end
        colormap(jet)
%         caxis([0 1])
        colorbar
    end
end
% plot bimodal response
for q=0:3
    figure
    for j =1+(q*16):(q+1)*16
        subplot(4,4,j-16*q);
        RR = reshape(RR_bi(j,:),size(XX));
        RR_vis = reshape(RR_uni_prop(j,:),size(XX));  %RR(1,:);
        RR_prop = reshape(RR_uni_vis(j,:),size(XX));  %RR(:,1);
        Ri = RR ./ (RR_vis + RR_prop);

        surf(XX,YY,flip(flip(RR,1),2)); 
        title(j)
%         axis([0.25 30.25 0.25 30.25]); %axis 'auto z'
        view([0 0 10]); shading interp
        if (j == 4), xlabel('Proprioceptive reliability (a.u.)'); end
        if (j == 4), ylabel('Visual reliability (a.u.) '); end
        if (j == 2), title(['\bf{bimodal resp, layer ' num2str(lay) '}']); end
        colormap(jet)
%         caxis([0 1])
        colorbar
    end
end

%%%%%%%%%%%%   sample units plot in paper
% units = [16 24 34 54]; % 
%  
% f = figure;
% set(f, 'Position', [100, 100, 800, 350]); % Adjust figure size and position
% 
% % Set margins and sizes for subplots
% margin_x = 0.02; % Horizontal margin
% margin_y = 0.02; % Vertical margin
% width = 0.2; % Width of each subplot
% height = 0.3; % Height of each subplot
% 
% for i = 1:4
%     j = units(i);
%     RR = reshape(RR_bi(j, :), size(XX));
%     RR_vis = reshape(RR_uni_prop(j, :), size(XX));
%     RR_prop = reshape(RR_uni_vis(j, :), size(XX));
%     Ri = RR ./ (RR_vis + RR_prop);
%     
%     % First row of subplots
%     left = 0.05 + (i - 1) * (width + margin_x);
%     bottom = 0.58; % First row position
%     ax1 = subplot(2, 4, i);
%     set(ax1, 'Position', [left, bottom, width, height]);
%     surf(XX, YY, flip(flip(Ri, 1), 2));
%     view([0 0 10]); shading interp;
%     colorbar();
%     title('Additivity Index', 'FontSize', 7);
%     if (i == 1), ylabel('Visual reliability (a.u.) ', 'FontSize', 7); end
%     
%     % Second row of subplots
%     bottom = 0.15; % Second row position
%     ax2 = subplot(2, 4, i + 4);
%     set(ax2, 'Position', [left, bottom, width, height]);
%     surf(XX, YY, flip(flip(RR, 1), 2));
%     view([0 0 10]); shading interp;
%     title('Bimodal Response', 'FontSize', 7);
%     if (i == 1), xlabel('Proprioceptive reliability (a.u.)', 'FontSize', 7); end
%     colormap(jet);
%     colorbar();
% end
% % 
% saveas(gcf, 'units_inverse_effectiveness.svg');

%% plot enhancement for different layers 
all_Re=[];
all_Ra=[];
f=figure;
set(f, 'Position', [80, 80, 650, 600]); % Adjust figure size and position
lay = 2;
for j = 1:64
    RR = reshape(ZA_bi{lay}(j,:),size(XX));
    RR_vis = reshape(ZA_uni_vis{lay}(j,:),size(XX));
    RR_prop = reshape(ZA_uni_prop{lay}(j,:),size(XX));
    Ra = (RR-(RR_vis+RR_prop))./(RR+(RR_vis+RR_prop)); % response additivity
    Re = (RR-max(RR_vis,RR_prop))./(RR+max(RR_vis,RR_prop)); % response enhancement

    all_Ra = [all_Ra; Ra(:)];
    all_Re = [all_Re; Re(:)];

    plot(Re*100, Ra*100, 'k.');
    xlim([-100 100])
    ylim([-100 100])
    hold on

end
xlabel('Response enhancement(%)'); 
ylabel('Response additivity(%)'); 
title(['\bf{Multi-sensory suppression, layer ' num2str(lay) '}']); 
saveas(gcf, 'enhancement&additivity_distribution_layer2.svg');
% Plot histograms for Ra and Re after the loop
f = figure;
set(f, 'Position', [80, 80, 650, 250]); % Adjust figure size and position

binEdges = -1:0.2:1;
h=histogram(-1*all_Ra,'BinEdges',binEdges); % 'Normalization', 'probability' ******reverse(*-1) the values for my plot adjustment
h.FaceColor = [0.75,0.44,1]; % RGB triplet for teal color 
h.EdgeColor = 'w'; % White edges
h.FaceAlpha = 0.8; % Transparency level
title('Ra');
% Move y-axis to the right
ax = gca; % Get current axes handle
ax.YAxisLocation = 'right'; % Set y-axis location to right
ylabel('N. trials');
xlim([binEdges(1), binEdges(end)]);
% saveas(gcf, 'response_additivity_histogram_layer2.svg');

f = figure;
set(f, 'Position', [80, 80, 650, 250]); % Adjust figure size and position
h =histogram(all_Re,'BinEdges',binEdges);
h.FaceColor = [0.75,0.44,1]; % RGB
h.EdgeColor = 'w'; 
h.FaceAlpha = 0.8; 
title('Re');
ylabel('N. trials');
xlim([binEdges(1), binEdges(end)]);
% saveas(gcf, 'response_enhancement_histogram_layer2.svg');
%% plot Amplification index   

all_x = [];
all_y = [];
figure

sz = 25;
c=0;
for q=0:3
    for j =1+(q*16):(q+1)*16

        RR = reshape(RR_bi(j,:),size(XX));
        RR_vis = reshape(RR_uni_prop(j,:),size(XX));  %RR(1,:);
        RR_prop = reshape(RR_uni_vis(j,:),size(XX));  %RR(:,1);
        amp_indx = (RR-max(RR_vis,RR_prop))./(RR+max(RR_vis,RR_prop)); 
        % they chose only enhancive neurons so we will consider only those units with positive amp_indx
        y = diag(amp_indx);
        pos_y=y(y>0)*100;
        x= diag(max(RR_vis,RR_prop));
        x_range = x(y>0);
        mask = pos_y < 70;
        x_range = x_range(mask);
        pos_y = pos_y(mask);


        all_x = [all_x; x_range];
        all_y = [all_y; pos_y];
%         scatter(x_range,pos_y,sz,'filled')
        hold on 

    end
end
scatter(all_x,all_y,sz,'filled',"b")
coefficients = polyfit(all_x, all_y, 1);
% x_line = linspace(min(all_x), max(all_x));
x_line = linspace(0, 1);
y_line = polyval(coefficients, x_line);
% y_line = max(y_line, 0); % Clip negative values to zero
plot(x_line, y_line, 'k-', 'LineWidth', 2)
ylabel('Amplification index(%) ');
xlabel('Dominant unimodal response'); ylim([-9,70]);
% saveas(gcf, 'amplification_index.svg');
%% plot Barplot for each neuron in uni and bi and sum response 
 
Resp_mat = zeros(64,4);
for i = 0:3
    figure % plot results (1-D)   
    for j = 1+(i*16):(i+1)*16
        subplot(4,4,j-16*i); 
        Resp_mat(j,1) = max(ZA_uni_prop{lay}(j,:));
        Resp_mat(j,2) = max(ZA_uni_vis{lay}(j,:));
        Resp_mat(j,3) = max(ZA_bi{lay}(j,:));
        Resp_mat(j,4) = max(ZA_uni_vis{lay}(j,:)) + max(ZA_uni_prop{lay}(j,:));
        x_plot = categorical(["P" "V" "VP" "V+P"]);
        x_plot = reordercats(x_plot,{'P' 'V' 'VP' 'V+P'});
        b = bar(x_plot, Resp_mat(j,:),'FaceColor',"flat");
        clr = [255 0 0; 0 255 0; 0 0 0; 128 128 128] / 255;
        b.CData = clr;
        title(num2str(j))
        if (j == 4), ylabel('Activity'); end
        hold on; 
   end
end
hold off
%samples plot in paper
chosen_units = [24 23 42 11];
for i= 1:4
    resp_mat =zeros(1,4);
    f = figure;
    set(f, 'Position', [80, 80, 400, 300]); % Adjust figure size and position
    j= chosen_units(i);
    resp_mat(j,1) = max(ZA_uni_prop{lay}(j,:));
    resp_mat(j,2) = max(ZA_uni_vis{lay}(j,:));
    resp_mat(j,3) = max(ZA_bi{lay}(j,:));
    resp_mat(j,4) = max(ZA_uni_vis{lay}(j,:)) + max(ZA_uni_prop{lay}(j,:));
    x_plot = categorical(["P" "V" "VP" "V+P"]);
    x_plot = reordercats(x_plot,{'P' 'V' 'VP' 'V+P'});
    b = bar(x_plot, resp_mat(j,:),'FaceColor',"flat");
    clr = [255 0 0; 0 255 0; 0 0 0; 128 128 128] / 255;
    ylabel('Activity');
    b.CData = clr;
%     fileName = sprintf('comparison_Activity_unit%d.svg', j);
%     saveas(gcf, fileName);
end
%% plot activity for each neuron in uni and bi and sum respons
x=1:11;
for i = 0:3
    figure % plot results (1-D)   
    for j = 1+(i*16):(i+1)*16
        subplot(4,4,j-16*i);
        Resp_mat = zeros(4,11);
        RR = reshape(RR_bi(j,:),size(XX));
        RR_vis = reshape(RR_uni_prop(j,:),size(XX));  
        RR_prop = reshape(RR_uni_vis(j,:),size(XX));  
        Resp_mat(1,:) = diag(RR_prop);
        Resp_mat(2,:) = diag(RR_vis);
        Resp_mat(3,:) = diag(RR);
        clr = [255 0 0; 0 255 0; 0 0 0] / 255;
        for y=1:3
            plot(x, Resp_mat(y,:),'Color', clr(y,:),'LineWidth', 2);
            hold on
        end
        title(num2str(j))
        if (j == 4), ylabel('Activity'); end
   end
end
hold off
% samples plot in paper
% x= 0:4:40; % equal to temp1 and temp2
% chosen_units= [42    53    45    18]
% for i= 1:4
%     j= chosen_units(i);
%     f = figure;
%     set(f, 'Position', [80, 80, 400, 300]); % Adjust figure size and position
%     Resp_mat = zeros(3,11);
%     RR = reshape(RR_bi(j,:),size(XX));
%     RR_vis = reshape(RR_uni_prop(j,:),size(XX));  
%     RR_prop = reshape(RR_uni_vis(j,:),size(XX));  
%     Resp_mat(1,:) = diag(RR_prop);
%     Resp_mat(2,:) = diag(RR_vis);
%     Resp_mat(3,:) = diag(RR);
%     clr = [255 0 0; 0 255 0; 0 0 0] / 255;
%     for y=1:3
%         plot(x, Resp_mat(y,:),'Color', clr(y,:),'LineWidth', 2);
%         hold on
%     end
%     title(num2str(j))
%     ylabel('Activity');
%     xlabel("Prop./Vis. position")
%     fileName = sprintf('Activity_unit%d.svg', j);
%     if (i==2), legend({"P","V","VP"}, 'Location', 'northeastoutside'); end
% %     lgd.ItemTokenSize = [10, 10];
%     saveas(gcf, fileName);
% end

%% plot distribution of response additivity for all units 
% 
% x=1:11;
% dist_additivity=zeros(64,11);
% dist_enhancement=zeros(64,11);
% % exceptunits = [3,5,20,28];% these 4 units have very large values for activation in order of 10e8!
% f = figure;
% set(f, 'Position', [80, 80, 600, 700]); % Adjust figure size and position
% for unit=1:64
%     RR = reshape(RR_bi(unit,:),size(XX));
%     RR_vis = reshape(RR_uni_prop(unit,:),size(XX));  
%     RR_prop = reshape(RR_uni_vis(unit,:),size(XX));
%     Ra = (RR-(RR_vis+RR_prop))./(RR+(RR_vis+RR_prop)); % response  additivity definition for enhancement plot
%     Re = (RR-max(RR_vis,RR_prop))./(RR+max(RR_vis,RR_prop)); % response enhancement
%     Ra_plot = flip(flip(Ra,1),2); % or rot90(Ri,2)  %% reorder additivity based on reliability not variance
%     Re_plot = flip(flip(Re,1),2); 
%     dist_additivity(unit,:)= diag(Ra_plot).*100;
%     dist_enhancement(unit,:)= diag(Re_plot).*100;
% 
%     subplot(2,1,1);
% %     if dist_additivity(unit,:)>0
%     scatter(x,dist_additivity(unit,:),12,'d',"filled")
%     hold on
% %     end
%     subplot(2,1,2);
% %     if dist_enhancement(unit,:)>0
%     scatter(x,dist_enhancement(unit,:),12,'d',"filled")
%     hold on
% %     end
% end
% errors_additivity = zeros(1,11);
% mean_points_additivity = zeros(1,11);
% for i =1:11
%     errors_additivity(1,i) = std(dist_additivity(:,i));%max(dist_additivity(i,:))-min(dist_additivity(i,:));%;
%     mean_points_additivity(1,i) = mean(dist_additivity(:,i));
% end
% 
% subplot(2,1,1);
% errorbar(x,mean_points_additivity,errors_additivity, '-o','MarkerSize', 6, 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'Black','LineWidth', 1.5);
% xlabel("Stimulus reliability (a.u.)");ylabel("Response additivity (%)");xlim([0 12]);%title("Distribution of response additivity for all units");xlim([0 12]);
% 
% errors_enhancement = zeros(1,11);
% mean_points_enhancement = zeros(1,11);
% for i =1:11
%     errors_enhancement(1,i) = std(dist_enhancement(:,i));
%     mean_points_enhancement(1,i) = mean(dist_enhancement(:,i));
% end
% subplot(2,1,2);
% errorbar(x,mean_points_enhancement,errors_enhancement, '-o','MarkerSize', 6, 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'Black','LineWidth', 1.5);
% xlabel("Stimulus reliability (a.u.)");ylabel("Response enhancement (%)");xlim([0 12]);%title("Distribution of response enhancemnt");xlim([0 12]);
% saveas(gcf, 'errorbar_EnhancementAdditivity.svg');
%%
% %%%%%% plot only positive additivity which mean enhanced respnses
% figure
% positive_dist_additivity = dist_additivity;
% positive_dist_additivity(positive_dist_additivity <= 0) = NaN;
% for unit = 1:64
%     scatter(var, positive_dist_additivity(:, unit) * 100, 12, 'd', "filled");
%     hold on;
% end
% positive_errors_additivity = zeros(1, 11);
% positive_mean_points_additivity = zeros(1, 11);
% for i = 1:11
%     positive_values = positive_dist_additivity(i, :);
%     positive_values = positive_values(~isnan(positive_values)); % Remove NaNs
%     if ~isempty(positive_values)
%         positive_errors_additivity(1, i) = std(positive_values); % Standard deviation
%         positive_mean_points_additivity(1, i) = mean(positive_values); % Mean
%     else
%         positive_errors_additivity(1, i) = NaN;
%         positive_mean_points_additivity(1, i) = NaN;
%     end
% end
% errorbar(var, positive_mean_points_additivity * 100, positive_errors_additivity * 100, '-o', 'MarkerSize', 6, 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'Black', 'LineWidth', 1.5);
% xlabel("stimulus variance");ylabel("Response Additivity(%)");title("Distribution of enhanced responses for all units")












