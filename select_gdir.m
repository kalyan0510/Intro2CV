## Copyright (C) 2021 Garigapati
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} select_gdir (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Garigapati <kalyanga@147dda75859f.ant.amazon.com>
## Created: 2021-05-24

% Gradient Direction
disp("Running...");
function result = select_gdir(gmag, gdir, mag_min, angle_low, angle_high)
    % TODO Find and return pixels that fall within the desired mag, angle range
    result = ((gmag >= mag_min) .* (gdir > angle_low) ).* (gdir < angle_high);
endfunction

pkg load image;

%% Load and convert image to double type, range [0, 1] for convenience
img = double(imread('octave.png')(:,:,1)) / 255.; 
imshow(img); % assumes [0, 1] range for double images

%% Compute x, y gradients
[gx gy] = imgradientxy(img, 'sobel'); % Note: gx, gy are not normalized

%% Obtain gradient magnitude and direction
[gmag gdir] = imgradient(gx, gy);
imshow(gmag / (4 * sqrt(2))); % mag = sqrt(gx^2 + gy^2), so [0, (4 * sqrt(2))]
imshow((gdir + 180.0) / 360.0); % angle in degrees [-180, 180]

%% Find pixels with desired gradient direction
my_grad = select_gdir(gmag, gdir, 1, 30, 60); % 45 +/- 15
imshow(my_grad);  % NOTE: enable after you've implemented select_gdir
