function F = root2d(x,m)
% Initialize variables.
filename = 'Rinfo.txt';
delimiter = ' ';

% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%f%[^\n\r]';

% Open the text file.
fileID = fopen(filename,'r');

% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this code. If an error occurs for a different file, try regenerating the code from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true,  'ReturnOnError', false);

% Close the text file.
fclose(fileID);

% Allocate imported array to column variable names
namee = dataArray{:, 1};
num = dataArray{:, 2};

% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me rawNumericColumns rawCellColumns;
% shift
s= num(1)
% length of RVE
LR=num(2)
% WAR = Aspect ratio w/h
WAR = num(4)
% volume fraction of mortar VM
VM= num(5)
% thickness ratio TR=t1/t2
TR= num(6)
F(1) = x(1)+x(3)-LR;
% F(2) = 2*(x(4)+x(2))-WR; %use this or the next one
F(2) = TR*x(2)-x(1);
F(3) = WAR*x(4)-x(3);
F(4) = VM*2*(x(4)+x(2))*LR-2*(x(2)*LR)-2*(x(1)*x(4));
