function [stf,pre_tar,misfit] = pld_liu(tar,egf,n0,nT,iter)
    % project Landweber deconvolution
    % f_n+1 = P(f_n + \tau G^T*(u-G*f_n))
    %% input:
    % tar: target event waveform
    % egf: empirical green function
    % nT: project axis
    % iter: iterations
    %% output:
    % stf: source time function
    % res: tar - stf*egf
    % misfit: norm(res,2)/norm(tar,2)
    stf = -1;
    res = -1;
    misfit = -1;
    epsilon = 0.05;
    ntar = length(tar);

    negf = length(egf);
    nfft = ceil(log2(ntar+negf));
    nstf = min(ntar,negf);
    D = zeros(2^nfft,1);% target
    GF = zeros(2^nfft,1);%egf
    for i = 1:ntar
       D(i) = tar(i); 
    end
    for i = 1:negf
       GF(i) = egf(i); 
    end
    %GT = [egf_new(1),flip(egf_new(2:end))];
    GFw = fft(GF);
    GTw = conj(GFw);% fft of GT, G* = fft(G(-t));
    Dw = fft(D);
    tau = max(abs(GFw));
    tau = 1/tau^2;
    index = [1:1:2^nfft]';
    fneww = zeros(2^nfft,1);
    fnewt = zeros(2^nfft,1);
    for it = 1:iter
       foldw = fneww;
       foldt = fnewt;
       resw = Dw - GFw.*foldw;
       misfit = norm(resw,2)/norm(D,2);
       if(misfit<epsilon)
          stf = foldt(1:nstf);
          break; 
       end
       gneww = foldw + tau*GTw.*resw;
       gnewt = real(ifft(gneww));
 
       fnewt = gnewt.*(index>=n0&index<=nT).*(gnewt>0); %project
       fneww = fft(fnewt);
    end
    if(stf==-1)
       stf = fnewt(1:nstf); 
    end
    pre_tar = conv(stf,GF);
    pre_tar = pre_tar(1:ntar);
    res = tar - pre_tar;
    misfit = norm(res,2)/norm(tar,2);
end

