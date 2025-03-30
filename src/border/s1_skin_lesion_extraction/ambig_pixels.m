% Suppl. 14

function ret = ambiguous_pixels(m)
if (m(2,2))==260
ret=[255,0,0];
else
    ret = m(2,2);
end
end