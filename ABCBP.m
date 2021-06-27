% 清空环境变量
clc
clear 

%读取数据
load newdata input output

%节点个数
inputnum=5;
outputnum=1;
hiddennum=14;

%训练和预测数据
input_train=input(1:1000,:)';
input_test=input(1001:1176,:)'; 
output_train=output(1:1000)';
output_test=output(1001:1176)';


%数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train); 
[inputn,mininput,maxinput,outputn,minoutput,maxoutput]=premnmx(input_train,output_train);
%net=newff(minmax(inputn,[5,1],{'tansig','purelin'},'trainlm');
net=newff(minmax(inputn),minmax(outputn),14);


% 初始化ABC参数
ABCOpts = struct( 'ColonySize',  2000, ...  
    'MaxCycles', 1000,...  
    'ErrGoal',   1e-20, ... 
    'Dim',       1, ... 
    'Limit',   50, ... 
    'lb',  -1000, ... 
    'ub',  1000, ... 
    'ObjFun' , 'JFC', ...
    'RunTime',1); 



GlobalMins=zeros(ABCOpts.RunTime,ABCOpts.MaxCycles);

for r=1:ABCOpts.RunTime
    
% Initialise population
Range = repmat((ABCOpts.ub-ABCOpts.lb),[ABCOpts.ColonySize ABCOpts.Dim]);
Lower = repmat(ABCOpts.lb, [ABCOpts.ColonySize ABCOpts.Dim]);
Colony = rand(ABCOpts.ColonySize,ABCOpts.Dim) .* Range + Lower;

Employed=Colony(1:(ABCOpts.ColonySize/2),:);


%evaluate and calculate fitness
ObjEmp=feval(ABCOpts.ObjFun,output_train,Employed);
FitEmp=calculateFitness(ObjEmp);

%set initial values of Bas
Bas=zeros(1,(ABCOpts.ColonySize/2));


GlobalMin=ObjEmp(find(ObjEmp==min(ObjEmp),end));
GlobalParams=Employed(find(ObjEmp==min(ObjEmp),end),:);

Cycle=1;
while ((Cycle <= ABCOpts.MaxCycles)),
    
    %%%%% Employed phase
    Employed2=Employed;
    for i=1:ABCOpts.ColonySize/2
        Param2Change=fix(rand*ABCOpts.Dim)+1;
        neighbour=fix(rand*(ABCOpts.ColonySize/2))+1;
            while(neighbour==i)
                neighbour=fix(rand*(ABCOpts.ColonySize/2))+1;
            end;
        Employed2(i,Param2Change)=Employed(i,Param2Change)+(Employed(i,Param2Change)-Employed(neighbour,Param2Change))*(rand-0.5)*2;
         if (Employed2(i,Param2Change)<ABCOpts.lb)
             Employed2(i,Param2Change)=ABCOpts.lb;
         end;
        if (Employed2(i,Param2Change)>ABCOpts.ub)
            Employed2(i,Param2Change)=ABCOpts.ub;
        end;
        
    end;  %for end

    ObjEmp2=feval(ABCOpts.ObjFun,output_train,Employed2);
    FitEmp2=calculateFitness(ObjEmp2);
    [Employed ObjEmp FitEmp Bas]=GreedySelection(Employed,Employed2,ObjEmp,ObjEmp2,FitEmp,FitEmp2,Bas,ABCOpts);

    %Normalize
    NormFit=FitEmp/sum(FitEmp);
    
    %%% Onlooker phase  
Employed2=Employed;
i=1;
t=0;
while(t<ABCOpts.ColonySize/2)
    if(rand<NormFit(i))
        t=t+1;
        Param2Change=fix(rand*ABCOpts.Dim)+1;
        neighbour=fix(rand*(ABCOpts.ColonySize/2))+1;
            while(neighbour==i)
                neighbour=fix(rand*(ABCOpts.ColonySize/2))+1;
            end;
         Employed2(i,:)=Employed(i,:);
         Employed2(i,Param2Change)=Employed(i,Param2Change)+(Employed(i,Param2Change)-Employed(neighbour,Param2Change))*(rand-0.5)*2;%find a new value in the neighborhood
         if (Employed2(i,Param2Change)<ABCOpts.lb)
             Employed2(i,Param2Change)=ABCOpts.lb;
         end;
        if (Employed2(i,Param2Change)>ABCOpts.ub)
            Employed2(i,Param2Change)=ABCOpts.ub;
         end;
    ObjEmp2=feval(ABCOpts.ObjFun,output_train,Employed2);
    FitEmp2=calculateFitness(ObjEmp2);
    [Employed ObjEmp FitEmp Bas]=GreedySelection(Employed,Employed2,ObjEmp,ObjEmp2,FitEmp,FitEmp2,Bas,ABCOpts,i);
   
   end; % end if
    
   i=i+1;
   if (i==(ABCOpts.ColonySize/2)+1) 
       i=1;
   end;   
end; %end while
    
    
    %%%Memorize Best
 CycleBestIndex=find(FitEmp==max(FitEmp));
 CycleBestIndex=CycleBestIndex(end);
 CycleBestParams=Employed(CycleBestIndex,:);
 CycleMin=ObjEmp(CycleBestIndex);
 
 if CycleMin<GlobalMin 
       GlobalMin=CycleMin;
       GlobalParams=CycleBestParams;
 end
 
 GlobalMins(r,Cycle)=GlobalMin;
 
 %% Scout phase
ind=find(Bas==max(Bas));
ind=ind(end);
if (Bas(ind)>ABCOpts.Limit)
Bas(ind)=0;
Employed(ind,:)=(ABCOpts.ub-ABCOpts.lb)*(0.5-rand(1,ABCOpts.Dim));%*2+ABCOpts.lb;
%message=strcat('burada',num2str(ind))
end;
ObjEmp=feval(ABCOpts.ObjFun,output_train,Employed);
FitEmp=calculateFitness(ObjEmp);



    fprintf('Cycle=%d ObjVal=%g\n',Cycle,GlobalMin);
    
    Cycle=Cycle+1;

end % End of ABC

end; %end of runs
x=GlobalMins;
%% 把最优初始阀值权值赋予网络预测
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP网络训练
%网络进化参数
net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;

%网络训练
[net,per2]=train(net,inputn,outputn);

%% BP网络预测
%数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error=test_simu-output_test;
RMSE=sqrt(sum((test_simu-output_test).^2)/176)
