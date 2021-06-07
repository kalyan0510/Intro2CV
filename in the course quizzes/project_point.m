% Project a point from 3D to 2D using a matrix operation

%% Given: Point p in 3-space [x y z], and focal length f
%% Return: Location of projected point on 2D image plane [u v]
disp("Running");
function p_img = project_point(p, f)
    %% TODO: Define and apply projection matrix
    perspective_matrix = [f 0 0 0; 0 f 0 0; 0 0 1 0];
    p_img = perspective_matrix * [p 1]';
    p_img = [p_img(1)/p_img(3) p_img(2)/p_img(3)];
endfunction

%% Test: Given point and focal length (units: mm)
p = [200 100 120];
f = 50;

disp(project_point(p, f));
