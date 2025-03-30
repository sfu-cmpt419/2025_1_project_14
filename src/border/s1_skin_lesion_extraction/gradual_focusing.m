% Suppl. 10

image_name=input('Enter the image name: ','s');
img = imread(image_name);
image = double(rgb2gray(img));

k=input('What are the number of clusters? ');
e=0.01;
 
    [row,col]=size(image);
    n=row*col;
    
    q=1;
    for i=1:row
        for j=1:col
              x(q)=image(i,j); 
              q=q+1;
        end
    end
    % centers of clusters (created randomly)
    c=row*rand(1,k);
    % fuzzy c-means
    m=2;
    p=1;
    objective_function=0;
    % membership degrees
    u=zeros(n,k);
    for i=1:n
       for j=1:k
           w=0;
           for q=1:k
               d=((x(i)-c(j)).^2)/((x(i)-c(q)).^2);
               w=w+d.^(2/(m-1));
           end
           u(i,j)=1/w;
       end
           objective_function=objective_function+((u(i,j)^2)*((x(i)-c(j)).^2));

    end
    
    done=true;
    while done
        % find the new cluster centers
        for j=1:k
            w1=0;
            w2=0;
            for i=1:n
                w1=w1+(u(i,j)^2)*x(i);
                w2=w2+u(i,j)^2;
            end
            c(j)=w1/w2; 
        end
        u1=u;
        % find new membership degrees
        for i=1:n
            for j=1:k
                w=0;
                for q=1:k
                     d=((x(i)-c(j)).^2)/((x(i)-c(q)).^2);
                     w=w+d.^(2/(m-1));
                end
               u(i,j)=1/w;
            end
               objective_function=objective_function+((u(i,j)^2)*((x(i)-c(j)).^2));
        end
        
        p=p+1;
        done=((u1-u).^2).^(1/2)<e;
        disp('DONE')
    end
   
    % find the ambiguity threshold
    
    % optimum threshold depending on a membership function (i.e; S-Function)
    threshold = optimumThreshold(image,u);
    disp('threshold')
    disp(threshold)
    
      
    for i=1:k
        non_ambiguous=0;
        nr=0;
        for j=1:n
            % labeling by using the maximum membership
            [l,pos]=max(u(j,:));
            if u(j,i)>=threshold
               im(j)=pos;
               non_ambiguous=non_ambiguous+x(j);
               nr=nr+1;
             
            elseif u(j,i)<threshold
                % -1 is used as a flag for the ambiguous pixel. If the
                % neighbourhood pixels equally belong to the different
                % clusters, the center pixel will be assigned to the 
                % cluster it belongs to based on the membership value
                   im(j)=-1*pos;
            end
            end
        end
                avg(i)=round(non_ambiguous/nr);
    q=1;
    n=1;
    image=zeros(row,col);
    
    for i=1:row
       for j=1:col
       image(i,j)=im(j);    
       end
    end
    
sz = size(image); 
% matrix showing the clusters each pixel belongs to based on the membership value
labeled_matrix = reshape(im,sz(2),sz(1)); 

% colour ambiguous pixels
new_img_pixels = nlfilter(labeled_matrix, [3,3], @ambiguous_pixels);
% check the neighbourhood of the ambiguous pixels to assign them to the appropriate 
% cluster
new_img = nlfilter(labeled_matrix, [3,3], @ambiguous_cluster);
% make sure no flag (negative pixel value) remain   
new_img(new_img<0) = new_img(new_img<0)*-1; 
 
% use below if more than 2-clusters
[a,b]=hist(new_img(:),unique(new_img));
thresh=max(new_img(:));  
bwImg = zeros(size(new_img));
bwImg(new_img >= thresh) = 0;
bwImg(new_img < thresh) = 1;  
output_img= bwImg';
imwrite(output_img,'segmentation.png','png');