% Suppl. 12

function ret = sFunction(x)
a=0;
b=0.5;
c=1;
if x<=a
    ret=0;
end

if a<x<=b
    ret=0.5*(x-a/b-a)^2;
end

if b<x<=c
    ret=1-0.5*(x-c/c-b)^2;
end

if x>c
    ret=1;
end
end