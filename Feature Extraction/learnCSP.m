function CSPMatrix = learnCSP(EEGSignals,classLabels)
%
%Input:
%EEGSignals: the training EEG signals, composed of 2 classes. These signals
%are a structure such that:
%   EEGSignals.x: the EEG signals as a [Ns * Nc * Nt] Matrix where
%       Ns: number of EEG samples per trial
%       Nc: number of channels (EEG electrodes)
%       nT: number of trials
%   EEGSignals.y: a [1 * Nt] vector containing the class labels for each trial
%   EEGSignals.s: the sampling frequency (in Hz)
%
%Output:
%CSPMatrix: the learnt CSP filters (a [Nc*Nc] matrix with the filters as rows)
%
%See also: extractCSPFeatures

%check and initializations
nbChannels = size(EEGSignals.x,2);
nbTrials = size(EEGSignals.x,3);
nbClasses = length(classLabels);

if nbClasses ~= 2
    disp('ERROR! CSP can only be used for two classes');
    return;
end

covMatrices = cell(nbClasses,1); %the covariance matrices for each class

%% Computing the normalized covariance matrices for each trial
trialCov = zeros(nbChannels,nbChannels,nbTrials);
for t=1:nbTrials
    E = EEGSignals.x(:,:,t)';                       %note the transpose
    EE = E * E';
    trialCov(:,:,t) = EE ./ trace(EE);
end
clear E;
clear EE;

%computing the covariance matrix for each class
for c=1:nbClasses      
    covMatrices{c} = mean(trialCov(:,:,EEGSignals.y == classLabels(c)),3); %EEGSignals.y==classLabels(c) returns the indeces corresponding to the class labels  
end

%the total covariance matrix
covTotal = covMatrices{1} + covMatrices{2};

%whitening transform of total covariance matrix
[Ut Dt] = eig(covTotal); %caution: the eigenvalues are initially in increasing order
eigenvalues = diag(Dt);
[eigenvalues egIndex] = sort(eigenvalues, 'descend');
Ut = Ut(:,egIndex);
P = diag(sqrt(1./eigenvalues)) * Ut';

%transforming covariance matrix of first class using P
transformedCov1 =  P * covMatrices{1} * P';

%EVD of the transformed covariance matrix
[U1 D1] = eig(transformedCov1);
eigenvalues = diag(D1);
[eigenvalues egIndex] = sort(eigenvalues, 'descend');
U1 = U1(:, egIndex);
CSPMatrix = U1' * P;