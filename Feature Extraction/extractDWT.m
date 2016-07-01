function X=extractDWT(x_train,startS,endS,wStep,wRange)
%  x_train  = input signal
%  startS   = from second
%  endS     = end second
%  wStep   = overlapping
%  wRange = window size
FS=128;

N=size(x_train,3);
sz=floor((endS-(startS+wRange))/wStep)+1;
X=zeros(sz*140,2);
cn=0;
for i=1:N
    
    for sig=startS:wStep:endS-wRange
        
        sW=sig*FS+1;
        eW=(sig+wRange)*FS;
        
        C3Sig=x_train(sW:eW,1,i);
        C4Sig=x_train(sW:eW,3,i);
        
        waveletFunction = 'db4';
        waveletLevel=3;
        [wCoe,L] = wavedec(C3Sig,waveletLevel,waveletFunction);
        C3D3 = detcoef(wCoe,L,3);   % Mu
        
        
        [wCoe,L] = wavedec(C4Sig,waveletLevel,waveletFunction);
        C4D3 = detcoef(wCoe,L,3);  % Mu
        
        cn=cn+1;
        % Mean of the absolute values
        X(cn,1)=sum(C3D3.^2)/numel(C3D3);
        X(cn,2)=sum(C4D3.^2)/numel(C4D3);
    end
end

end
