tic;

N = 128;
M = 200;
  
[Ac Bc] = deal(complex( gones(N,N,M, 'single'),0));

gfor ii = 1:M
    Ac(:,:,ii) = fft2(Bc(:,:,ii));
gend

Ac = single(Ac);

toc / n

tic;

N = 128;
M = 200;
  
[Ac Bc] = deal(complex(ones(N,N,M, 'single'),0));

for ii = 1:M
    Ac(:,:,ii) = fft2(Bc(:,:,ii));
end

toc / n