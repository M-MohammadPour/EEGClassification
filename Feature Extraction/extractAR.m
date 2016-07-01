function X=extractAR(x_train,AROrder,startS,endS,wStep,wRange)

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
        
        c3= arburg(C3Sig, AROrder);
        c4= arburg(C4Sig, AROrder);
        cn=cn+1;
        X(cn,1)=sum(c3.^2)/numel(c3);
        X(cn,2)=sum(c4.^2)/numel(c4);
        
    end
end

end

