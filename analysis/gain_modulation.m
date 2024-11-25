clear all;
load('net_refframe_and_multisensory_5lay_50000_64');

%% gain field / receptive field analysis
temp1 = -45:45; % retinal hand position
temp2 = 1*(-45:5:45); %% random set of associated proprioceptive hand position
temp3 = 0; % reference frame transformation angle
temp4 = (0:5/18:5).^2+0.1;
n1 = length(temp1); n2 = length(temp2); n3 = length(temp3);
%ff = 50;

Ainput = [repmat(temp1',n2,1) reshape(repmat(temp2',1,n1)',n1*n2,1) repmat(temp3,n2*n1,1)]';
AVarVis = repmat(2.5.^2,1,n1*n2); % visual variance
% AVarVis = reshape(repmat(temp4',1,n1)',1,n1*n2);
AVarPro = repmat(3.5.^2,1,n1*n2); % proprioceptive variance
% AVarPro = reshape(repmat(temp4',1,n1)',1,n1*n2);
AVarEye = repmat(3.5.^2,1,n1*n2); % eye position variance
% AVarEye = reshape(repmat(temp4',1,n1)',1,n1*n2);
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
ZA = sim(net,Pdana); % simulate trained network


for lay = 1:2,
   
    figure % plot results (2-D)
    [XX,YY] = meshgrid(temp2, temp1);
    for j = 17:32,
        subplot(4,4,j-16);
        zz = reshape(ZA{lay}(j,:)',size(XX));
        surf(YY,XX,zz);
         title(['\bf{unit' num2str(j) '}']);
        axis([-45 45 -45 45 0 1]); axis 'auto z'
        view([0 0 1]); shading interp
        if (j == 1), ylabel('Proprioceptive position (deg)'); end
        if (j == 1), xlabel('Visual position (deg)'); end
        if (j == 2), title(['\bf{Receptive field changes with proprioception, layer ' num2str(lay) '}']); end
        colormap(jet)
        caxis([0 1])
        colorbar
    end

    f=figure;
    set(f, 'Position', [80, 80, 300, 250]);
    for i = 1:n2,
        for j = 1:net.layers{lay}.size,
            ACT{1}(i,j) = nanmean(ZA{lay}(j,(i-1)*n1+(1:n1)));   % mean for one prop value i in all range vis(1:n1)
        end
    end

    ACTg{1} = (ACT{1}(n2,:)-ACT{1}(1,:))./(ACT{1}(n2,:)+ACT{1}(1,:)); %calculate gain 
    hist(ACTg{1}(find(abs(ACTg{1}(:))<1.1)),-1:.25:1)
    title(['\bf{Gain changes with Prop Position, layer ' num2str(lay) '}']);

    fileName = sprintf('Viseye_gainmodulation_layer%d.svg', lay);
%     saveas(gcf, fileName);

end

% plot for sample units
f=figure;
set(f, 'Position', [80, 80, 300, 250]);
lay = 1;
j = 57;
zz = reshape(ZA{lay}(j,:)',size(XX));
surf(YY,XX,zz);
axis([-45 45 -45 45 0 1]); axis 'auto z'
% axis([-45 45 0 25 0 1]); axis 'auto z'
view([0 0 1]); shading interp
colormap(jet)
% caxis([0 1])
title("SIL layer")
colorbar
fileName = sprintf('Viseye_unit_Rf_layer%d.svg', lay);
% saveas(gcf, fileName);

f=figure;
set(f, 'Position', [80, 80, 300, 250]);
lay = 2;
j = 30;
zz = reshape(ZA{lay}(j,:)',size(XX));
surf(YY,XX,zz);
axis([-45 45 -45 45 0 1]); axis 'auto z'
% axis([-45 45 0 25 0 1]); axis 'auto z'
view([0 0 1]); shading interp
colormap(jet)
caxis([0 1])
title("MSL layer")
colorbar
fileName = sprintf('Viseye_unit_Rf_layer%d.svg', lay);
% saveas(gcf, fileName);

