% Downsample an image
1;  % Octave script guard

function img_d = downsample(img)
    % TODO: img_d = ? (pick alternate rows, cols: 1, 3, 5, ...)
    img_d = img(1:2:size(img)(1), 1:2:size(img)(2));
end

function img_bd = blur_downsample(img)
    % TODO: img_bd = ? (blur by 5x5 gaussian, then downsample)
    % filter = [1 2 1; 2 4 2; 1 2 1]/16;
    filter = fspecial('gaussian', [5 5]);
    img = uint8(conv2(img, filter, 'same'));
    img_bd = img(1:2:size(img)(1), 1:2:size(img)(2));
end

% Test code:
pkg load image;

img = imread('octave.png')(:,:,1);
imshow(img);
size(img)

img_d = downsample(img);    % 1/2 size
img_d = downsample(img_d);  % 1/4 size
img_d = downsample(img_d);  % 1/8 size

img_bd = blur_downsample(img);     % 1/2 size
img_bd = blur_downsample(img_bd);  % 1/4 size
img_bd = blur_downsample(img_bd);  % 1/8 size
figure, imshow(img_d);
figure, imshow(img_bd);
%imshow(imresize(img_d, size(img)));   % view downsampled image in original size
%imshow(imresize(img_bd, size(img)));  % compare with blurred & downsampled
