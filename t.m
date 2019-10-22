clc
clear
Displ=[]
AR=3
LR=1 % LR: length of RVE (=1 fixed), 
WR=1 %WR: width of RVE,
TR=1 %  TR: t1/t2,
sh=0.1 %shifting distance of top layer of brick
VM=0.1 % volume fraction of mortar
Eb=1 %elastic modulus of brick
Em=0.5 %elastic modulus of mortar
Pb=0.3 %poisson ratio of brick
Pm=0.3 %poisson ratio of mortar
    fileID3 = fopen('x.csv','w');
    fprintf(fileID3,'LR,WR,AR,w,h,t1,t2,sh,Eb,Em,Pb,Pm,VM,TR,');
    fprintf(fileID3,'Young,El_MaxP,MaxP,El_MaxY_HI,MaxY_HI,El_MaxY_VI,MaxY_VI\n');
        fileID4 = fopen('y.csv','w');
    fprintf(fileID4,'LR,WR,AR,w,h,t1,t2,sh,Eb,Em,Pb,Pm,VM,TR,');
    fprintf(fileID4,'Young,El_MaxP,MaxP,El_MaxY_HI,MaxY_HI,El_MaxY_VI,MaxY_VI\n');
a = [13	0.4	0.5	1.963787827]
    Em=0.5;
     AR = a(1);
     sh=a(2);
    VM=a(3);
    TR=a(4);
    
                      fileID = fopen('Rinfo.txt','w');
       fprintf(fileID,'sh %2.4f\nLR %2.0f\nWR %2.4f\nAR %2.0f\nVM %2.4f\nTR %2.4f\n',sh,LR,WR,AR,VM,TR);
    fclose(fileID)
    fun = @root2d;
    x0 = [0.1,0,0,0];
    x = fsolve(fun,x0);
    t1 = x(1); %distance between brick or interface horizontal thickness
    t2 = x(2); %distance between brick or interface vertical thickness
    w = x(3); % width of the brick 
    h = x(4); %height of brick
    WR=2*(h+t2); %width of RVE
        if (t1<0) | (t2<0) | (h<0)|(w<0)
        fun = @root2db;
            x0 = [0.1,0,0,0];
    x = fsolve(fun,x0);
        t1 = x(1); %distance between brick or interface horizontal thickness
    t2 = x(2); %distance between brick or interface vertical thickness
    w = x(3); % width of the brick 
    h = x(4); %height of brick
    WR=2*(h+t2); %width of RVE
    end 
 fileID2 = fopen('ABQinfo.txt','w');    
fprintf(fileID2,'%2.0f\n%2.8f\n%2.4f\n%2.8f\n%2.8f\n%2.8f\n%2.8f\n%2.4f\n%2.4f\n%2.4f\n%2.4f\n%2.4f\n',LR, WR,AR,w,h,t1,t2,sh,Eb,Em,Pb,Pm) %increase the accuracy to %2.8f
%    fprintf(fileID2,'LR %2.0f\nWR %2.4f\nAR %2.0f\nw  %2.4f\nh  %2.4f\nt1 %2.4f\nt2 %2.4f\nsh %2.4f\nEb %2.4f\nEm %2.4f\nPb %2.4f\nPm %2.4f\n',LR, WR,row,w,h,t1,t2,sh,Eb,Em,Pb,Pm)
    fclose(fileID2)
    !abaqus cae noGui=x.py
% x results printing
    results=importdata('result.txt')
    fprintf(fileID3,'%2.0f,%2.8f,%2.4f,%2.8f,%2.8f,%2.8f,%2.8f,%2.8f,%2.0f,%2.4f,%2.8f,%2.8f,%2.8f,%2.8f,%2.8f',LR, WR,AR,w,h,t1,t2,sh,Eb,Em,Pb,Pm)
    fprintf(fileID3,'%2.2f,%2.8f,%2.4f,%2.8f',VM,TR,results(1))
    fprintf(fileID3,'%3.0f,%2.8f,%3.0f,%2.8f,%3.0f,%2.8f\n',results(2), results(3),results(4),results(5),results(6),results(7))
        !abaqus cae noGui=y.py
% y results printing
    results=importdata('result.txt')
    fprintf(fileID4,'%2.0f,%2.8f,%2.4f,%2.8f,%2.8f,%2.8f,%2.8f,%2.8f,%2.0f,%2.4f,%2.8f,%2.8f,%2.8f,%2.8f,%2.8f',LR, WR,AR,w,h,t1,t2,sh,Eb,Em,Pb,Pm)
    fprintf(fileID4,'%2.2f,%2.8f,%2.4f,%2.8f',VM,TR,results(1))
    fprintf(fileID4,'%3.0f,%2.8f,%3.0f,%2.8f,%3.0f,%2.8f\n',results(2), results(3),results(4),results(5),results(6),results(7))
        
    
    fclose(fileID3)   
    fclose(fileID4)

% figure
% rectangle('position', [0 0 LR WR],'FaceColor',[0 .2 .5])
%rectangle('position', [t1/2 t2/2 w h],'FaceColor',[0 .5 .5])
%rectangle('position', [0 3*t2/2+h w-sh h],'FaceColor',[0 .5 .5])
%rectangle('position', [LR-sh 3*t2/2+h sh h],'FaceColor',[0 .5 .5])
%axis([-0.1 1.5 -0.1 1.5]);  