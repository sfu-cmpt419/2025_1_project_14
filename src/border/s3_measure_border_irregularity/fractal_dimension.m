% Suppl 4

% input: the segmented image
function [fractal]=fractal_dimesion(input)
segmentation = input;

e = edge(segmentation,'canny');

% form a grid of blocks across the image
% check if the edge occupies any of the blocks
% flag the occupied block and record it in boxCount
% store both size of blocks (numBlocks) and number of occupied boxes
% (boxCount) in a table
Nx = size(e,1);
Ny = size(e,2);

for numBlocks = 1:25
    
    sizeBlocks_x = floor(Nx./numBlocks);
    sizeBlocks_y = floor(Ny./numBlocks);
    
    flag = zeros(numBlocks,numBlocks);
    for i = 1:numBlocks
        for j = 1:numBlocks
            xStart = (i-1)*sizeBlocks_x + 1;
            xEnd   = i*sizeBlocks_x;
            
            yStart = (j-1)*sizeBlocks_y + 1;
            yEnd   = j*sizeBlocks_y;
            
            block = e(xStart:xEnd, yStart:yEnd);
            % if any part of the block is true, set (mark) the flag
            flag(i,j) = any(block(:)); 
        end
    end
    boxCount = nnz(flag);
    table(numBlocks,1) = numBlocks;
    table(numBlocks,2) = boxCount;
end


% the raw data and best fit line
x = table(:,1); % numBlocks
y = table(:,2); % boxCount

% Hausdorff dimension
x2 = log(x);
y2 = log(y);

p2 = polyfit(x2,y2,1);
BestFit2 = polyval(p2,x2);

figure(1)
hold on
grid on
plot(x2,y2,'bo','LineWidth',1)
plot(x2,BestFit2, 'b-','LineWidth',2)
xlabel('Number of blocks, log N','FontSize',12)
ylabel('Box Count, log N(s)'    ,'FontSize',12)

fractal = p2(:,1)