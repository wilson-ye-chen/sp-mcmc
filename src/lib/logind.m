function li = logind(x, lb, ub, A, b)
% li = logind(x, lb, ub, A, b) implements the log-indicator function.
%
% Date: April 16, 2018

    if ~isempty(lb) && any(x < lb)
        li = -inf;
        return
    end
    if ~isempty(ub) && any(x > ub)
        li = -inf;
        return
    end
    if ~isempty(A) && any(A * x > b)
        li = -inf;
        return
    end
    li = 0;
end
