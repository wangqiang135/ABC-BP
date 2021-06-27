 xlsfile='总实验数据.xlsx';
 pm10=xlsread('总实验数据.xlsx','F1:F1177');
 NO2=xlsread('总实验数据.xlsx','G1:G1177');
 SO2=xlsread('总实验数据.xlsx','H1:H1177');
 CO=xlsread('总实验数据.xlsx','I1:I1177');
 O3=xlsread('总实验数据.xlsx','J1:J1177');
 input=[pm10,NO2,SO2,CO,O3];
 output=xlsread('总实验数据.xlsx','E1:E1177');
 save newdata input output;