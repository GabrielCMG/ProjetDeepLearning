%%

Fe = 256;
N = 4096;
T = N/Fe;

t = 0:T/N:T-T/N;



for i = 1:5
    f = (rand([1, 1])+0.2)*5;
    A = 1;
    s = A*cos(2*pi*f*t);
    sB = s + wgn(1, 4096, 10);
    subplot(5, 2, 2*i-1)
    plot(t, s)
    subplot(5, 2, 2*i)
    plot(t, sB)
end

%%

for i = 1:1000
    f = (rand([1, 1])+0.2)*5;
    A = 1;
    s = A*cos(2*pi*f*t);
    sB = s + wgn(1, 4096, 2);
    sigList(i, :) = s;
    sigListNoise(i, :) = sB;
end

csvwrite('label1f.csv', sigList);
csvwrite('train1f.csv', sigListNoise);