%%

Fe = 256;   %frequence d'echantillonage
N = 4096;   %Cardinal du signal
T = N/Fe;   %Temps d'observation

t = 0:T/N:T-T/N;

nFreq = 1;  %Nombre de frequence d'un signal (nombre entier strictement positif)

%Generation des signaux avec nFreq frequence
for i = 1:1000
    fList = (rand([nFreq, 1])+0.2)*5;   %tirage aléatoirede la frequence
    A = 1;  %Amplitude associe a chaque frequence
    s = zeros(1, 4096);
    for j=1:nFreq
        s = s + A*cos(2*pi*fList(j)*t); %Generation frequence par frequence
    end
    sB = awgn(s, 5, 'measured');    %Ajout d'un bruit blanc gaussien
    sigList(i, :) = s;
    sigListNoise(i, :) = sB;
end

csvwrite('label3f.csv', sigList);   %Enregistrement du signal non bruite
csvwrite('train3f.csv', sigListNoise);  %Enregistrement du signal bruite