disp("Running");

function ssd = ssd(x,y)
  a = x-y;
  ssd = sum(a.*a);
endfunction
% Find best match
function best_x = find_best_match(patch, strip)
    % TODO: Find patch in strip and return column index (x value) of topleft corner
    best = [1 Inf];
    for i = [1:(size(strip)(2)-size(patch)(2)+1)]
      cur_ssd = ssd(patch, strip(:, i:(size(patch)(2)+i-1)));
      if cur_ssd < best(2)
        best = [i cur_ssd];
      endif
    endfor
    best_x = best(1);
endfunction

pkg load image;

% Test code:

%% Load images
left = imread('left.png');
right = imread('right.png');
figure, imshow(left);
figure, imshow(right);

%% Convert to grayscale, double, [0, 1] range for easier computation
left_gray = double(rgb2gray(left)) / 255.0;
right_gray = double(rgb2gray(right)) / 255.0;

%% Define image patch location (topleft [row col]) and size
patch_loc = [120 170];
patch_size = [100 100];

%% Extract patch (from left image)
patch_left = left_gray(patch_loc(1):(patch_loc(1) + patch_size(1) - 1), patch_loc(2):(patch_loc(2) + patch_size(2) - 1));
figure, imshow(patch_left);

%% Extract strip (from right image)
strip_right = right_gray(patch_loc(1):(patch_loc(1) + patch_size(1) - 1), :);
figure, imshow(strip_right);

%% Now look for the patch in the strip and report the best position (column index of topleft corner)
best_x = find_best_match(patch_left, strip_right);
disp(best_x);
patch_right = right_gray(patch_loc(1):(patch_loc(1) + patch_size(1) - 1), best_x:(best_x + patch_size(2) - 1));
figure, imshow(patch_right);
