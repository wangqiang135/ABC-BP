function ObjVal = JFC(output_train,Employed)
error=output_train'-Employed;
a=error.*error;
   ObjVal = a; 

