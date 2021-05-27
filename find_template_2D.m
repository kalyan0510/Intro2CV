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
## @deftypefn {} {@var{retval} =} find_template_2D (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Garigapati <kalyanga@147dda75859f.ant.amazon.com>
## Created: 2021-05-24
disp("Running...");
% Find template 2D
% NOTE: Function definition must be the very first piece of code here!
function [yIndex xIndex] = find_template_2D(template, img)
    % TODO: Find template in img and return [y x] location
    % NOTE: Turn off all output from inside the function before submitting!
    a= normxcorr2(template, img);
    [yIndex, xIndex] = find (a == max (a(:)))
    yIndex = yIndex - size(template,1)+1;
    xIndex = xIndex - size(template,2)+1;
endfunction

pkg load image; % AFTER function definition

% Test code:
tablet = imread('octave.png');
imshow(tablet);
glyph = tablet(75:165, 150:185);
imshow(glyph);

[y x] = find_template_2D(glyph, tablet);
disp([y x]); % should be the top-left corner of template in tablet
