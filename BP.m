%读取数据
load newdata input output


%训练预测数据
input_train=input(1:1000,:)';
input_test=input(1001:1176,:)';
output_train=output(1:1000)';
output_test=output(1001:1176)';



%数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);
[inputn,mininput,maxinput,outputn,minoutput,maxoutput]=premnmx(input_train,output_train); %对p和t进行字标准化预处理 %对p和t进行字标准化预处理 
%net=newff(minmax(inputn,[5,1],{'tansig','purelin'},'trainlm');
net=newff(minmax(inputn),minmax(outputn),10);

net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.001;
%net.trainParam.show=NaN

%网络训练
net=train(net,inputn,outputn);

%数据归一化
inputn_test = tramnmx(input_test,mininput,maxinput);

an=sim(net,inputn);

test_simu=postmnmx(an,minoutput,maxoutput);

error=test_simu-output_train;

plot(error)

BPoutput=mapminmax('reverse',an,outputps);
  p=output_train;
    q=inputn_test;
    n=length(q);
    e=sqrt(sum((p-q).^2)/n)
k=error./output_train;
hold on
plot(k)
