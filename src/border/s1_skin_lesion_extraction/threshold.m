% Suppl. 11

function ret = optimumThreshold(I,m)
[r c]=size(I);
image_size=r*c;
img=I(:);

% membership matrix
for i=1:image_size
membershipMatrix(i)=max(m(i,:));
end

imageVector=img(:);
uniqueImageVector=unique(imageVector);
repetition_all_imagevector=histc(imageVector,uniqueImageVector);
graylevels=numel(uniqueImageVector);


membershipMatrixVector=membershipMatrix(:);
unique_membershipMatrixVector=unique(membershipMatrixVector);
repetition_all_membershipmatrix=histc(membershipMatrixVector,unique_membershipMatrixVector);

% measure ultrafuzziness

for i=1:graylevels
membership_function_value=sFunction(membershipMatrixVector(i));
% alpha = 2
membership_upper=(membership_function_value)^0.5; 
membership_lower=(membership_function_value)^2;
membership_upper_lower_diff=membership_upper-membership_lower;
% repetition (frequency) of the grey level
repetition=repetition_all_imagevector(uniqueImageVector==imageVector(i)); 
hist_multiply_difference=repetition*membership_upper_lower_diff;
linear_index_of_fuzziness(i)=(1/image_size)*hist_multiply_difference;
end

ret = max(linear_index_of_fuzziness(:));

end