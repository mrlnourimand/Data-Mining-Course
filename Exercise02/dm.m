%Load in image file numbers.png.
img=imread('numbers.png');%Read in an image.
imshow(img);              %Show an image.
size(img);
gsimg=rgb2gray(img);      %convert to grayscale image.
size(gsimg);
imshow(gsimg);
bw=im2bw(gsimg)          %convert to black&white (binary) image.
bw=not(bw)               % change back an foregreound.

row=5; %Row index.
col=5;  %Column index.

%Crude way to extract individual numbers from image.
%Resulting images are not of same size.
delta=28.8; %size of an individual image. Note, non an integer.
n=1:10;
stop=round(n*delta);
start=round(n*delta+1);
start(end)=[]
start=[1 start]
num1=bw(start(row):stop(row),start(col):stop(col)); %A number from position (row,column)
imshow(num1)

row=1; %Row index.
col=5;  %Column index.

num2=bw(start(row):stop(row),start(col):stop(col)); %A number form position (row,column)
figure,
imshow(num2)

row=1; %Row index.
col=6;  %Column index.

num3=bw(start(row):stop(row),start(col):stop(col)); %A number form position (row,column)
figure,
imshow(num3)
