 xlsfile='��ʵ������.xlsx';
 pm10=xlsread('��ʵ������.xlsx','F1:F1177');
 NO2=xlsread('��ʵ������.xlsx','G1:G1177');
 SO2=xlsread('��ʵ������.xlsx','H1:H1177');
 CO=xlsread('��ʵ������.xlsx','I1:I1177');
 O3=xlsread('��ʵ������.xlsx','J1:J1177');
 input=[pm10,NO2,SO2,CO,O3];
 output=xlsread('��ʵ������.xlsx','E1:E1177');
 save newdata input output;