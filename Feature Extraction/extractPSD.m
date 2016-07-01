function X=extractPSD(x_train,startS,endS,wStep,wRange)
Fs=1;   % sampling frequency (seconds)

FS=128;

N=size(x_train,3);
sz=floor((endS-(startS+wRange))/wStep)+1;
X=zeros(sz*FS,2);
cn=0;
for i=1:N
    
    for sig=startS:wStep:endS-wRange
        
        sW=sig*FS+1;
        eW=(sig+wRange)*FS;
        
        C3Sig=x_train(sW:eW,1,i);
        C4Sig=x_train(sW:eW,3,i);
        
        [pxx3, ~] = pwelch(C3Sig,[],[],[],Fs);
        [pxx4, ~] = pwelch(C4Sig,[],[],[],Fs);
        
        cn=cn+1;
        X(cn,1)=sum(pxx3);
        X(cn,2)=sum(pxx4);
        
    end
end

end


