% im=imread('left.png')
% size(im(:,:,1))
% im = im(:,:,1)
% edg = edge(im, 'canny')
I  = imread('circuit.tif');
% size(I)/50
BW = edge(imrotate(I,50,'crop'),'canny');
[H,T,R] = hough(BW);
size(H)
P  = houghpeaks(H,4);
imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot(T(P(:,2)),R(P(:,1)),'s','color','white');