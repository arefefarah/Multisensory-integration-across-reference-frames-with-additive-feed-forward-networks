%
% creates a user-defined 3-layer neural network
% for multi-sensory integration in 1-D
%
%
%=============================================

% clear;
% warning off;

a0 = 0; % train network
a1 = 1; % analyze network (general)
a2 = 0; % specific multi-sensory integration analyses
xp = 1; % 0: multi-sensory integration
        % 1: noisy reference frame transformation
        % 2: both
a3 = 0; % spiking network equivalent

if a0 == 1,
%% creating training set
%----------------------
N = 50000; % length of training set
input(1,:) = 15*(randn(1,N)); % random set of retinal hand position
input(2,:) = input(1,:) + 10*(randn(1,N)); 
input(3,:) = input(2,:) + 10*(randn(1,N)); % random set of associated proprioceptive hand position
VarVis = (abs(1*(randn(1,N)+2))+1).^2; % visual variance
VarPro = (abs(1*(randn(1,N)+3))+1).^2; % proprioceptive variance
VarEye = (abs(1*(randn(1,N)+4))+1).^2; % eye position variance
% VarTot = VarVis.*VarPro./(VarVis+VarPro);
% output = VarTot.*(input(1,:)./VarVis + input(2,:)./VarPro); % spatial target position
if xp == 1
    VarTot = VarVis + VarEye;
    output = input(1,:) + input(2,:);
elseif xp == 2;
    VarTot = (VarVis+VarEye).*VarPro./(VarVis+VarEye+VarPro);
    output = VarTot.*((input(1,:)+input(2,:))./(VarVis+VarEye) + input(3,:)./VarPro); % spatial target position
end

oMax = max(abs(output));
vMax = max(abs(input(1,:)));
pMax = max(abs(input(2,:)));
if xp == 2, eMax = max(abs(input(3,:))); end

% network variables
x = -75:2:75; % preferred directions of input units (visual hand position coding)
Ni = length(x); % number input units (map)
Ne = 76; % number proprioceptive hand position coding input units
slope = 2*(rand(1,Ne)-.5);
offset = 20*(rand(1,Ne)-.5);
if xp == 2,
    slope2 = 2*(rand(1,Ne)-.5);
    offset2 = 20*(rand(1,Ne)-.5);
end
Nh = 64; % number hidden layer units
% No = length(x); % number population output units
No = 1;

% network input
%--------------
% variability coding in activation map
ff = 50;
tunVis = exp(-(repmat(x',1,N)-repmat(input(1,:),Ni,1)).^2./10.^2./2);
ampVis = ff./VarVis;
actVis = repmat(ampVis, Ni, 1).*tunVis;
Pd{1,1} = poissrnd(actVis)/ff; % activations of 1-D retinal map

tunEye = poslin(repmat(offset2',1,N) + repmat(slope2',1,N).*repmat(input(2,:),Ne,1)./eMax);
ampEye = ff./VarEye;
actEye = repmat(ampEye, Ne, 1).*tunEye;
Pd{2,1} = poissrnd(actEye)/ff; % eye position
if xp == 2,
    tunPro = poslin(repmat(offset',1,N) + repmat(slope',1,N).*repmat(input(3,:),Ne,1)./pMax);
    ampPro = ff./VarPro;
    actPro = repmat(ampPro, Ne, 1).*tunPro;
    Pd{3,1} = poissrnd(actPro)/ff; % proprioceptive coding
    
end
Ai = [];
Q = 0;
TS = 1;
VV = [];
TV = [];

% network output
% tunOut = exp(-(repmat(x',1,N)-repmat(output,No,1)).^2./10.^2./2);
ampOut = 1./VarTot;
% ampOut = 1*ones(size(VarTot));
% actOut = repmat(ampOut, Ni, 1).*tunOut;
Or(1) = range(output); Or(2) = range(VarTot);
actOut = [output/Or(1); VarTot/Or(2)];
%only var as output
%only mean as output
actOut = [output/Or(1)];
Tl{1,1} = actOut; % 1-D spatial output map

%% general network definition
%---------------------------
net = network; % create a network

net.numInputs = length(Pd); % number of physically different inputs
net.numLayers = 3; % number of layers (not including the input)
net.biasConnect = [0; 0; 0]; % bias enable of layers (0: no, 1: yes)
net.inputConnect = [ones(1, net.numInputs); zeros(1, net.numInputs); zeros(1, net.numInputs)]; 
% inputs are connected to layer indicated by "1"
net.layerConnect = [0 0 0; 1 0 0; 0 1 0]; % interconnection of layers (row: origin
                               %                            line: target)
net.outputConnect = [1 1 1]; % outputs are layers indicated by "1" (for simulation only)
net.targetConnect = [0 0 1]; % train target is the layer indicated by "1" (for training only)

% different potential training algorithms
net.trainFcn = 'trainrp'; % resilient back-prop
% net.trainFcn = 'trainscg'; % scaled conjugate gradient back-prop
% net.trainFcn = 'trainoss'; % one step secant back-prop
% net.trainFcn = 'traincgb'; % Powell-Beale conjugate gradient back-prop
% net.trainFcn = 'traingdx'; % gradient descent with momentum and adaptive learning back-prop
% net.trainFcn = 'traincgp'; % Conjugate gradient backpropagation with Polak-Ribiére updates

net.adaptFcn = 'trainb'; % batch training
net.performFcn = 'mse'; % error function for training performance monitoring

% specific network parameters
limV = max(ampVis);
limP = max(ampPro);
if xp == 2, limE = max(ampEye); end
limO = max(ampOut);
net.inputs{1}.size = Ni; % target direction
net.inputs{1}.range = repmat([-limV limV], net.inputs{1}.size, 1);
net.inputs{2}.size = Ne; % eye position coding
net.inputs{2}.range = repmat([-limP limP], net.inputs{2}.size, 1);
if xp == 2,
    net.inputs{3}.size = Ne; % eye position coding
    net.inputs{3}.range = repmat([-limE limE], net.inputs{2}.size, 1);
end
net.layers{1}.size = Nh; % number of hidden layer units
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.size = Nh; % number of population output units
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.size = No; % number of population output units
net.layers{3}.transferFcn = 'purelin';
net.trainParam.goal = 1e-5; % desired error
net.trainparam.epochs = 1e4;  % maximum training steps
net.trainParam.show = 1; % shows training progress at ever step
net.trainParam.min_grad = 1e-7; % minimum allowed gradient

% initialize network (i.e. weights)
net.initFcn = 'initlay';
for j = 1:net.numInputs,
    net.inputWeights{1,j}.initFcn = 'rands'; % random initial input weights
    net.inputWeights{1,j}.delays = 0;
end
for j = 1:net.numLayers,
    net.layers{j}.initFcn = 'initwb';
%     net.biases{j}.initFcn = 'rands';
    for k = 1:net.numLayers,
        net.layerWeights{j,k}.initFcn = 'rands'; % random initial layer weights
        net.layerWeights{j,k}.delays = 0;
    end
end
net = init(net);

% save user data
%--------------
net.userdata.Or = Or;
net.userdata.ff = ff;
net.userdata.Pd = Pd;
net.userdata.Tl = Tl;
net.userdata.offset = offset;
net.userdata.slope = slope;
net.userdata.x = x;
if xp == 2,
    net.userdata.offset2 = offset2;
    net.userdata.slope2 = slope2;
end

%% traing the net
%---------------
tic;

[net,TR] = train(net,Pd,Tl); % batch training 
% (use 'adapt' for sequential training)
% [net,TR,Y,E] = minibatchTrain(net,Pd,Tl,1000)
set(gca,'Xscale','log');
disp(['Training with ' net.trainFcn ' needed '...
    num2str(toc,'%10.3f') ' s'])

%% simulate and save network
%--------------------------
Ntest = 5000; % length of test set

% test input/output
Tinput(1,:) = 15*(randn(1,Ntest)); % random set of retinal hand position
Tinput(2,:) = Tinput(1,:) + 10*(randn(1,Ntest)); 
Tinput(3,:) = Tinput(2,:) + 10*(randn(1,Ntest)); % random set of associated proprioceptive hand position
TVarVis = (abs(1*(randn(1,Ntest)+2))+1).^2; % visual variance
TVarPro = (abs(1*(randn(1,Ntest)+3))+1).^2; % proprioceptive variance
TVarEye = (abs(1*(randn(1,Ntest)+4))+1).^2;
% TVarTot = TVarVis.*TVarPro./(TVarVis+TVarPro);
% Toutput = TVarTot.*(Tinput(1,:)./TVarVis + Tinput(2,:)./TVarPro); % spatial target position
if xp == 1,
    TVarTot = TVarVis + TVarEye;
    Toutput = Tinput(1,:) + Tinput(2,:);
elseif xp == 2,
    TVarTot = (TVarVis+TVarEye).*TVarPro./(TVarVis+TVarEye+TVarPro);
    Toutput = TVarTot.*((Tinput(1,:)+Tinput(2,:))./(TVarVis+TVarEye) + Tinput(3,:)./TVarPro); % spatial target position
end

TtunVis = exp(-(repmat(x',1,Ntest)-repmat(Tinput(1,:),Ni,1)).^2./10.^2./2);
TampVis = ff./TVarVis;
TactVis = repmat(TampVis, Ni, 1).*TtunVis;
Pdtest{1,1} = poissrnd(TactVis)./ff; % activations of 1-D retinal map
TtunPro = poslin(repmat(offset',1,Ntest) + repmat(slope',1,Ntest).*repmat(Tinput(2,:),Ne,1)./pMax);
TampPro = ff./TVarPro;
TactPro = repmat(TampPro, Ne, 1).*TtunPro;
Pdtest{2,1} = poissrnd(TactPro)./ff; % eye position coding
if xp == 2,
    TtunEye = poslin(repmat(offset2',1,Ntest) + repmat(slope2',1,Ntest).*repmat(Tinput(3,:),Ne,1)./eMax);
    TampEye = ff./TVarEye;
    TactEye = repmat(TampEye, Ne, 1).*TtunEye;
    Pdtest{3,1} = poissrnd(TactEye)/ff; % proprioceptive coding
end

% test net
Z = sim(net,Pdtest); % simulate trained network
figure
[m, b, r] = postreg(Z{3}(1,:)*Or(1), Toutput);
figure
[m, b, r] = postreg(Z{3}(2,:)*Or(2), TVarTot);


net.userdata.epochs = TR; % save training progress in userdata structure of network
save('net_refframe_and_multisensory_5lay_50000_64.mat')

end

