1;  % Octave script guard
% Compute Integral Image
function I = compute_integral(img)
    % TODO: Compute I such that I(y, x) = sum of img(1, 1) to img(y, x)
    I = img;
    sz = size(img);
    for i=2:sz(1)
      I(i,:,:) += I(i-1,:,:);   
    endfor
    for i=2:sz(2)
      I(:,i,:) += I(:,i-1,:);    
    endfor
end

% Test code:
pkg load image;

img = im2double(imread('octave.png'));
%%imshow(img);

%% Compute integral
I = compute_integral(img);
imshow(I / max(I(:)));

%% Compare sum over an image window
x1 = 150; y1 = 100;
x2 = 350; y2 = 200;

disp(sum(img(y1:y2, x1:x2)(:)));
disp(I(y2, x2) - I(y1 - 1, x2) - I(y2, x1 - 1) + I(y1 - 1, x1 - 1));

I(y2, x2)
I(y1-1, x2)
I(y2, x1-1)
I(y1-1, x1-1)

