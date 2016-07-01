function features = extractCSP(EEGSignals, CSPMatrix, nbFilterPairs)
%
%extract features from an EEG data set using the Common Spatial Patterns (CSP) algorithm
%
%Input:
%EEGSignals: the EEGSignals from which extracting the CSP features. These signals
%are a structure such that:
%   EEGSignals.x: the EEG signals as a [Ns * Nc * Nt] Matrix where
%       Ns: number of EEG samples per trial
%       Nc: number of channels (EEG electrodes)
%       nT: number of trials
%   EEGSignals.y: a [1 * Nt] vector containing the class labels for each trial
%   EEGSignals.s: the sampling frequency (in Hz)
%CSPMatrix: the CSP projection matrix, learnt previously (see function learnCSP)
%nbFilterPairs: number of pairs of CSP filters to be used. The number of
%   features extracted will be twice the value of this parameter. The
%   filters selected are the one corresponding to the lowest and highest
%   eigenvalues
%
%Output:
%features: the features extracted from this EEG data set 
%   as a [Nt * (nbFilterPairs*2 + 1)] matrix, with the class labels as the
%   last column   
%
%initializations

nbTrials = size(EEGSignals.x,3);
features = zeros(nbTrials, 2*nbFilterPairs+1);
Filter = CSPMatrix([1:nbFilterPairs (end-nbFilterPairs+1):end],:);

%extracting the CSP features from each trial
for t=1:nbTrials    
    %projecting the data onto the CSP filters    
    projectedTrial = Filter * EEGSignals.x(:,:,t)';    
    
    %generating the features as the log variance of the projected signals
    variances = var(projectedTrial,0,2);    
    for f=1:length(variances)
        features(t,f) = log(1+variances(f));
    end
end