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
## @deftypefn {} {@var{retval} =} zsuv (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Garigapati <kalyanga@147dda75859f.ant.amazon.com>
## Created: 2021-05-24

function retval = zsuv (t)
  stdVal = std(t);
  retval = (t - mean (t)) ./ stdVal;
  if stdVal==0
    retval=zeros(size(t));
  endif
endfunction
