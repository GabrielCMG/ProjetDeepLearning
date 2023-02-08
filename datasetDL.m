%%

Fe = 256;
N = 4096;
T = N/Fe;

t = 0:T/N:T-T/N;

nFreq = 3;

for i = 1:1000
    fList = (rand([nFreq, 1])+0.2)*5;
    A = 1;
    s = zeros(1, 4096);
    for j=1:nFreq
        s = s + A*cos(2*pi*fList(j)*t);
    end
    sB = awgn(s, -7, 'measured');
    sigList(i, :) = s;
    sigListNoise(i, :) = sB;
end

csvwrite('label3f.csv', sigList);
csvwrite('train3f.csv', sigListNoise);