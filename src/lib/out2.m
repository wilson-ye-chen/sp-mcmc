function o2 = out2(f)
% o2 = out2(f) returns the second output of a function.
%
% Input:
% f  - handle to a function that returns at least two outputs.
%
% Output:
% o2 - the second output.
%
% Date: December 12, 2017

    [~, o2] = f();
end
