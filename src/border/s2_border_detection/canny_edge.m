% Suppl 5

D = '/Users/abder/Downloads/test-smoothed';
S = dir(fullfile(D,'*.png')); 
for k = 1:numel(S)
    F = fullfile(D,S(k).name);
    disp(F);
    I = imread(F);
    I4 = edge(I,'Canny');
    imwrite(I4,strcat('/Users/abder/Downloads/test-smoothed-edges/',S(k).name));
end