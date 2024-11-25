
clear;
load('net_refframe_and_multisensory_5lay_50000_64')
load('weights.mat')
Nvis = 25;
Nprop = 25;
Ntest = Nvis*Nprop; % length of test set
%% for receptive field with only changing position and two fixed variance

temp1 = -45:3.7:45; % retinal hand positions
temp2 = temp1; %+ 10 * (randn(1,Nvis)); % random set of associated proprioceptive hand position
temp3 = 0; % reference frame transformation angle
n1 = length(temp1); n2 = length(temp2); n3 = length(temp3);
ff = 50;
Ainput = [repmat(temp1',n2,1) reshape(repmat(temp2',1,n1)',n1*n2,1) repmat(temp3,n2*n1,1)]';
%%%% for var =4 just remove to the power of 2
AVarVis = (abs(1*(ones(1,Ntest)+2))+1); % visual variance
AVarEye = (abs(1*(ones(1,Ntest)+1))+1).^2;  % eye position variance
AVarPro = (abs(1*(ones(1,Ntest)+2))+1).^2; % proprioceptive variance


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
ZAbi = sim(net,Pdana); % simulate trained network
lay = 2;
%% plot  receptive field / neuron acitivity of units in different layers for cue reliability
[XX, YY] = meshgrid(-45:3.7:45, -45:3.7:45);
lay = 2;
var=16;
% for q=0:3
%     figure
%     for j =1+(q*16):(q+1)*16
%         subplot(4,4,j-16*q);
%         surf(XX, YY, reshape(ZAbi{lay}(j, :), size(XX)));
%         title(j)
%         axis([-45 43 -45 43 0 1]);
%         view([0 0 10]); shading interp
%         axis 'auto z';
%         view([0 0 1]); 
%         shading interp;
%         colormap(jet);
%         colorbar
%     end
% end

% lay 1 : unit 57, 41 
%  lay 2 : unit 35 ,41 
% for var =16

% f = figure;
% set(f, 'Position', [80, 80, 800, 450]); % Increase figure size for better clarity
% 
% margin_x = 0.07; %  horizontal margin
% margin_y = 0.07; %  vertical margin
% width = 0.25; % Adjust width of each subplot
% height = 0.35; % Adjust height of each subplot   
% 
% lay = 1; 
% ideal_units = [31, 47, 45];
% 
% for i = 1:
%     j = ideal_units(i);
%     left = 0.05 + (i - 1) * (width + margin_x);
%     bottom = 0.58; % First row position
%     ax1 = subplot(2, 3, i);
%     set(ax1, 'Position', [left, bottom, width, height]);
%     surf(XX, YY, reshape(ZAbi{lay}(j, :), size(XX)));
%     axis([-45 43 -45 43 0 1]); 
%     axis 'auto z';
%     view([0 0 1]); 
%     shading interp;
%     colormap(jet);
%     title(['\bf{unit ' num2str(j) ' layer ' num2str(lay) '}']);
% %         colorbar;
%     if i ~= 3, xticks([]); yticks([]); end
% end
% 
% lay = 2; 
% ideal_units = [30, 47, 45];
% 
% for i = 1:3
%     j = ideal_units(i);
%     bottom = 0.08; % Second row position
%     left = 0.05 + (i - 1) * (width + margin_x);
%     ax1 = subplot(2, 3, i + 3);
%     set(ax1, 'Position', [left, bottom, width, height]);
%     surf(XX, YY, reshape(ZAbi{lay}(j, :), size(XX)));
%     axis([-45 43 -45 43 0 1]); 
%     axis 'auto z';
%     view([0 0 1]); 
%     shading interp;
%     colormap(jet);
%     title(['\bf{unit ' num2str(j) ' layer ' num2str(lay) '}']);
% %         colorbar;
%     if i == 3 , xlabel('Proprioceptive position (deg)');ylabel('Visual position (deg)'); end
%     if i ~= 3, xticks([]); yticks([]); end
% end
%%
%%% plot for paper
for i = 1:2
    lay=i;
    j = 41;
    figure
    surf(XX, YY, reshape(ZAbi{lay}(j, :), size(XX)));
    axis([-45 43 -45 43 0 1]); 
    axis 'auto z';
    view([0 0 1]); 
    shading interp;
    colormap(jet);
    title(['\bf{unit ' num2str(j) ' layer ' num2str(lay) '}']);
    colorbar;
    xlabel('Proprioceptive position (deg)');ylabel('Visual position (deg)'); 
    fileName = sprintf('unit_Activity_layer%d.svg', i);
%     saveas(gcf, fileName);
end


%% cue reliability and fixed position to test visual and prop weight in linear Regression model
% % Response to visual stimuli
% 
% SV = [1 2.5 4.5 6];
% OutSim1 = cell(4,1);
% OutSim2 = cell(4,1);
% inbi = cell(4,1);
% 
% ktemp11 = -45:3.7:45;
% ktemp22 = ktemp11 + 10 * (randn(1,Nvis));
% ktemp33 = ktemp22 + 10 * (randn(1,Nvis));
% kinput(1,:) = repmat(ktemp11,1,Nprop);
% t = repmat(ktemp22,Nvis,1);
% kinput(2,:) = (t(:))';
% kinput(3,:) = repmat(ktemp33,1,Nprop);
% 
% kVarPro = ((3.5).^2).*ones(1,Ntest); % proprioceptive variance
% % kinput(3,:) = kinput(2,:) + 10 *(randn(1,Ntest));
% kVarEye = (abs(1*(ones(1,Ntest)+1))+1).^2; % eye position variance
% 
% for i = 1:length(SV)
%     kVarVis = (abs(1*(ones(1,Ntest)+1))+SV(i)).^2; % visual variance
%     kVarTot = (kVarVis+kVarEye).*kVarPro./(kVarVis+kVarEye+kVarPro);
%     Koutput = kVarTot.*((kinput(1,:)+kinput(3,:))./(kVarVis+kVarEye) + kinput(2,:)./kVarPro); % spatial target position
%     ktunVis = exp(-(repmat(x',1,Ntest)-repmat(kinput(1,:),Ni,1)).^2./10.^2./2);
%     kampVis = ff./kVarVis;
%     kactVis = repmat(kampVis, Ni, 1).*ktunVis;
%     ktest{1,1} = (kactVis)./ff; % activations of 1-D retinal map
%     KtunPro = poslin(repmat(offset',1,Ntest) + repmat(slope',1,Ntest).*repmat(kinput(2,:),Ne,1)./pMax);
%     kampPro = ff./kVarPro;
%     kactPro = repmat(kampPro, Ne, 1).*KtunPro;
%     ktest{2,1} = (kactPro)./ff; % proprioceptive coding
%     if xp == 2,
%         KtunEye = poslin(repmat(offset2',1,Ntest) + repmat(slope2',1,Ntest).*repmat(kinput(3,:),Ne,1)./eMax);
%         kampEye = ff./kVarEye;
%         kactEye = repmat(kampEye, Ne, 1).*KtunEye;
%         ktest{3,1} = (kactEye)/ff; % proprioceptive coding
%     end
%     Zt = sim(net,ktest); % simulate trained network
% %         figure
% %         [m, b, r] = postreg(Zt{3}(1,:)*Or(1), Koutput);
% %         figure
% %         [m, b, r] = postreg(Zt{3}(2,:)*Or(2), kVarTot);
%     OutSim1{i} = Zt{1};  %save output of layers 1 & 2 for future
%     OutSim2{i} = Zt{2};
% end
% inbi = [kinput(1,:)' kinput(2,:)' kinput(3,:)'];
% svbi = SV;
% 
%  %% unisensory responses
% 
% % Response to proprioceptive stimuli
% Nvis = 25;
% Nprop = 25;
% Ntest = Nvis*Nprop; % length of test set
% SV = 6.75;
% Outprop = cell(2,1);
% % Outprop2 = cell(1,1);
% ktemp11 = -45:3.7:45;
% kinput(1,:) = repmat(ktemp11,1,Nprop);
% ktemp22 = ktemp11 + 10 * (randn(1,Nvis));
% t = repmat(ktemp22,Nvis,1);
% kinput(2,:) = (t(:))';
% kVarPro = (abs(1*(ones(1,Ntest)+2))+1).^2; % proprioceptive variance
% 
% ktemp33 = ktemp22 + 10 * (randn(1,Nvis));
% kinput(3,:) = repmat(ktemp33,1,Nprop);
% % kinput(3,:) = kinput(2,:) + 10 *(randn(1,Ntest));
% kVarEye = (abs(1*(ones(1,Ntest)+1))+1).^2; % eye position variance
% 
% for i = 1:length(SV)
%     kVarVis = (abs(1*(ones(1,Ntest)+1))+SV(i)).^2; % visual variance
%     kVarTot = (kVarVis+kVarEye).*kVarPro./(kVarVis+kVarEye+kVarPro);
%     Koutput = kVarTot.*((kinput(1,:)+kinput(3,:))./(kVarVis+kVarEye) + kinput(2,:)./kVarPro); % spatial target position
%     ktunVis = exp(-(repmat(x',1,Ntest)-repmat(kinput(1,:),Ni,1)).^2./10.^2./2);
%     kampVis = ff./kVarVis;
%     kactVis = repmat(kampVis, Ni, 1).*ktunVis;
%     ktest{1,1} = (kactVis)./ff; % activations of 1-D retinal map
%     KtunPro = poslin(repmat(offset',1,Ntest) + repmat(slope',1,Ntest).*repmat(kinput(2,:),Ne,1)./pMax);
%     kampPro = ff./kVarPro;
%     kactPro = repmat(kampPro, Ne, 1).*KtunPro;
%     ktest{2,1} = (kactPro)./ff; % proprioceptive coding
%     if xp == 2,
%         KtunEye = poslin(repmat(offset2',1,Ntest) + repmat(slope2',1,Ntest).*repmat(kinput(3,:),Ne,1)./eMax);
%         kampEye = ff./kVarEye;
%         kactEye = repmat(kampEye, Ne, 1).*KtunEye;
%         ktest{3,1} = (kactEye)/ff; % proprioceptive coding
%     end
%     
%     Zt = sim(net,ktest); % simulate trained network
% %         figure
% %         [m, b, r] = postreg(Zt{3}(1,:)*Or(1), Koutput);
% %         figure
% %         [m, b, r] = postreg(Zt{3}(2,:)*Or(2), kVarTot);
%     Outprop{1} = Zt{1};
%     Outprop{2} = Zt{2};
% end
% inprop = [kinput(1,:)' kinput(2,:)' kinput(3,:)'];
% svprop = SV;
% 
% % Response to visual stimuli
% Nvis = 25;
% Nprop = 25;
% Ntest = Nvis*Nprop; % length of test set
% SV = [1 2.5 4.5 6];
% Outvis1 = cell(4,1);
% Outvis2 = cell(4,1);
% invis = cell(4,1);
% ktemp11 = -45:3.7:45;
% kinput(1,:) = repmat(ktemp11,1,Nprop);
% ktemp22 = ktemp11 + 10 * (randn(1,Nvis));
% t = repmat(ktemp22,Nvis,1);
% kinput(2,:) = (t(:))';
% kVarPro = ((8.6).^2).*ones(1,Ntest); % proprioceptive variance
% 
% ktemp33 = ktemp22 + 10 * (randn(1,Nvis));
% kinput(3,:) = repmat(ktemp33,1,Nprop);
% % kinput(3,:) = kinput(2,:) + 10 *(randn(1,Ntest));
% kVarEye = (abs(1*(ones(1,Ntest)+1))+1).^2; % eye position variance
% 
% for i = 1:length(SV)
%     kVarVis = (abs(1*(ones(1,Ntest)+1))+SV(i)).^2; % visual variance
%     kVarTot = (kVarVis+kVarEye).*kVarPro./(kVarVis+kVarEye+kVarPro);
%     Koutput = kVarTot.*((kinput(1,:)+kinput(3,:))./(kVarVis+kVarEye) + kinput(2,:)./kVarPro); % spatial target position
%     ktunVis = exp(-(repmat(x',1,Ntest)-repmat(kinput(1,:),Ni,1)).^2./10.^2./2);
%     kampVis = ff./kVarVis;
%     kactVis = repmat(kampVis, Ni, 1).*ktunVis;
%     ktest{1,1} = (kactVis)./ff; % activations of 1-D retinal map
%     KtunPro = poslin(repmat(offset',1,Ntest) + repmat(slope',1,Ntest).*repmat(kinput(2,:),Ne,1)./pMax);
%     kampPro = ff./kVarPro;
%     kactPro = repmat(kampPro, Ne, 1).*KtunPro;
%     ktest{2,1} = (kactPro)./ff; % proprioceptive coding
%     if xp == 2,
%         KtunEye = poslin(repmat(offset2',1,Ntest) + repmat(slope2',1,Ntest).*repmat(kinput(3,:),Ne,1)./eMax);
%         kampEye = ff./kVarEye;
%         kactEye = repmat(kampEye, Ne, 1).*KtunEye;
%         ktest{3,1} = (kactEye)/ff; % proprioceptive coding
%     end
%     % 
%     % Or(1) = range(Koutput); Or(2) = range(kVarTot);
%     % test net
%     Zt = sim(net,ktest); % simulate trained network
% %         figure
% %         [m, b, r] = postreg(Zt{3}(1,:)*Or(1), Koutput);
% %         figure
% %         [m, b, r] = postreg(Zt{3}(2,:)*Or(2), kVarTot);
%     Outvis1{i} = Zt{1};
%     Outvis2{i} = Zt{2};
% 
% end
% invis = [kinput(1,:)' kinput(2,:)' kinput(3,:)'];
% svvis = SV;
%     save('weights.mat','svbi','inbi','OutSim1','OutSim2','inprop','Outprop',...
%         'svvis','invis','Outvis1','Outvis2','Ntest')


%% plot weights as a function of reliability 
% load('weights.mat')
% weights = zeros(64*4,2);
% for i = 1 : 4
%     for j = 1 : 64
%         xreg = [ones(Ntest,1) Outvis2{i}(j,:)' Outprop{2}(j,:)'];
%         yreg = OutSim2{i}(j,:)';
%         b = regress(yreg,xreg);
%         weights((i-1)*64+j,:) = b(2:3);
%     end
%     
%     
% end
% q = weights(:,1) > 100; 
% weights(q,1) = 0;
% q = weights(:,2) > 100; 
% weights(q,2) = 0;
% visual= zeros(64,4);
% vest = visual;
% %     figure; hold on
% for i = 1:64
%     temp = weights(i:64:4*64,1);
% %         plot(svbi,temp(4:-1:1))
%     visual(i,:) = weights(i:64:4*64,1);
% end
% %     figure; hold on
% for i = 1:64
%     temp = weights(i:64:4*64,2);
% %         plot(svbi,temp(4:-1:1))
%     vest(i,:) = weights(i:64:4*64,2);
% end
% wvp = zeros(4,2);
% for i = 1:4
%     wvp(i,1) = mean(weights((i-1)*64+1:i*64,1)./weights((i-1)*64+1:i*64,2));
%     wvp(i,2) = std(weights((i-1)*64+1:i*64,1)./weights((i-1)*64+1:i*64,2));
% end
% %     figure ; errorbar(svbi, wvp(:,1), wvp(:,2))
% vvms = zeros(4,4);
% for i = 1 : 4
%     q = visual(:,i) < 5 & visual(:,i) > -5;t = visual(q);
%     temp1 = visual(:,i);
%     vvms(1,i) = mean(temp1(q));
%     vvms(2,i) = std(temp1(q));
%     q = vest(:,i) < 1 & vest(:,i) > -1;
%     temp2 = vest(:,i);
%     vvms(3,i) = mean(temp2(q));
%     vvms(4,i) = std(temp2(q));
% 
% 
% end
% x_range = 1./(svbi+2).^2; %this is reliability
% 
% f = figure;
% set(f, 'Position', [80, 80, 400, 250]); 
% errorbar(1:4,vvms(1,4:-1:1),vvms(2,4:-1:1)./2); ylabel(["Visual weights"]); xlabel(['Visual reliability (a.u.)']);ylim([0 1]);xlim([0.5 4.5])
% % saveas(gcf, 'visual_weight.svg');
% f = figure;
% set(f, 'Position', [80, 80, 400, 250]); 
% errorbar(1:4,vvms(3,4:-1:1),vvms(4,4:-1:1)./2); ylabel(["Proprioceptive weights"]); xlabel(['Visual reliability (a.u.)']);ylim([0 1]);xlim([0.5 4.5])
% saveas(gcf, 'Proprioceptive_weight.svg');


%% calculate R^2

% R2 = zeros(64,4);
% mean_r2 = zeros(4,1);
% for i=1:4,
%     for j=1:64,
%         Rbi = OutSim2{i}(j,:)';
%         w = [vvms(1,i) vvms(3,i)];
%         xxx = [Outvis2{i}(j,:)' Outprop{2}(j,:)']';
%         Weighted_bothRuni = w*xxx;
%         [m, b, r] = postreg(Weighted_bothRuni',Rbi);
%         R2(j,i) = r;
%     end
%     mean_r2(i,1) = mean(R2(:,i));
% end



%% plot gain field
%     ACT = cell(3,1)
%     figure % plot gain modulation of mean and std activation
%     for i = 1:3,
%         i
%         for j = 1:net.layers{lay}.size,
%             j
%             ACT{1}(i,j) = mean(mean(reshape(ZAbi{lay}(j,(100*(i-1)*n1)+(1:100*n1)),n1,100)));
%             ACT{2}(i,j) = mean(std(reshape(ZAbi{lay}(j,(100*(i-1)*n1)+(1:100*n1)),n1,100)));
%         end
%     end
%     ACTg{1} = (ACT{1}(3,:)-ACT{1}(1,:))./ACT{1}(2,:);
%     ACTg{2} = (ACT{2}(3,:)-ACT{2}(1,:))./ACT{2}(2,:);
%     subplot(2,1,1)
%     hist(ACTg{1}(:),-10:.25:10)
%     title(['\bf{Gain changes with Prop Var, layer ' num2str(lay) '}']);
%     xlabel('Noise modulation gain - mean RF')
%     subplot(2,1,2)
%     hist(ACTg{2}(:),-10:.25:10)
%     xlabel('Noise modulation gain - variance RF')


%     lay = 2
%     figure
%     for i = 1:n2,
%         for j = 1:net.layers{lay}.size,
%             ACT{1}(i,j) = nanmean(ZAbi{lay}(j,(i-1)*n1+(1:n1)));
%         end
%     end
%     ACTg{1} = (ACT{1}(n2,:)-ACT{1}(1,:))./(ACT{1}(n2,:)+ACT{1}(1,:));
% %     hist(ACTg{1}(find(abs(ACTg{1}(:))<1.1)),-1:.25:1)
%     hist(ACTg{1}(:),-1:.25:1)
%     title(['\bf{Gain changes with Prop Position, layer ' num2str(lay) '}']);
%     xlabel('Proprioceptive gain modulation - mean RF');

   