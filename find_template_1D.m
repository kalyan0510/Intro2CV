disp("Running...")
% Find template 1D
% NOTE: Function definition must be the very first piece of code here!

function retval = zsuv (t)
  stdVal = std(t);
  retval = (t - mean (t)) ./ stdVal;
  if stdVal==0
    retval=zeros(size(t));
  endif
endfunction


function index = find_template_1D(t, s)
  tsize = size(t,2);
  nt = zsuv(t);
  index = 0;maxVal = -1/eps;
  for i = 1:size(s,2)-size(t,2)+1
    cval = zsuv(s(i:i+(size(t,2)-1)))*nt';
    cond = cval > maxVal;
    index = cond * i + ~cond * index;
    maxVal = cond * cval + ~cond * maxVal;
  endfor
endfunction

pkg load image; % AFTER function definition

% Test code:
s = [-1 0 0 1 1 1 0 -1 -1 0 1 0 0 -1];
t = [1 1 0];
disp('Signal:'), disp([1:size(s, 2); s]);
disp('Template:'), disp([1:size(t, 2); t]);

index = find_template_1D(t, s);
disp('Index:'), disp(index);