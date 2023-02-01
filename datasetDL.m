%%

Fe = 40;
N = 4096;
T = N/Fe;

t = 0:T/N:T-T/N;



for i = 1:5
    fList = rand([3, 1])*10;
    AList = [1 1 1];
    s = AList(1)*cos(2*pi*fList(1)*t) + AList(2)*cos(2*pi*fList(2)*t) + AList(3)*cos(2*pi*fList(3)*t); 
    sB = s + wgn(1, 4096, 0);
    subplot(5, 2, 2*i-1)
    plot(t, s)
    subplot(5, 2, 2*i)
    plot(t, sB)
end

%%

for i = 1:1000
    fList = rand([3, 1])*10;
    AList = [1 1 1];
    s = AList(1)*cos(2*pi*fList(1)*t) + AList(2)*cos(2*pi*fList(2)*t) + AList(3)*cos(2*pi*fList(3)*t); 
    sB = s + wgn(1, 4096, 0);
    sigList(i, :) = s;
    sigListNoise(i, :) = sB;
end

csvwrite('label.csv', sigList);
csvwrite('train.csv', sigListNoise);