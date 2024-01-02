%
%

%Numbers from 1 to 100.
t1=1:100


%Dimensions of "Matrix t". Even a scalar is matrix in Matlab (Matrix laboratory)
l=length(t1)
d=size(t1)

%Transpose of t.
tp=t1'
size(tp)

%100 samples from 50Hz sinusoidal signal sampled at frequency of 8192Hz.
f=50
F=8192
y=sin(2*pi*t1*(f/F))
plot(t1,y)

%Another signal into the same plot.

y2=cos(2*pi*t1*(f/F))

plot(t1,y)
hold on
plot(t1,y2,'g')

%Dimensions to the axes.
plot(t1,y)
hold on
plot(t1,y2,'g')
xlabel('Time instant');
ylabel('Amplitude')

%Legends to the graphs.
plot(t1,y)
hold on
plot(t1,y2,'g')
xlabel('Time instant');
ylabel('Amplitude')
legend('Sine','Cosine');


%difference of signals y and y2.
d1=y2-y;
plot(d1);
