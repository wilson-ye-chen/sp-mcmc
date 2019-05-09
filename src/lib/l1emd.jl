# Usage: l1emd.jl MAT_IN MAT_OUT
# Computes the Wasserstein distance between pairs of point sets stored in
# MAT files. The output is also stored in a MAT file. The input file must
# contain three variables:
#
# XPts - m-by-2 matrix of points,
# YPts - n-by-2 matrix of points,
# SegY - k-by-2 matrix of segments,
#
# where m, n, and k are positive integers. The output file contains:
#
# emd  - k-by-1 column vector of Wasserstein distance estimates.
#
# Date: January 8, 2018

using Clp
using SteinDiscrepancy: wassersteindiscrete
using MAT

# Command line arguments
matIn = ARGS[1];
matOut = ARGS[2];

# Read in MATLAB variables
file = matopen(matIn);
XPts = read(file, "XPts");
YPts = read(file, "YPts");
SegY = convert(Array{Int}, read(file, "SegY"));
close(file);

# Create solver ahead of time
solver = Clp.ClpSolver(LogLevel = 4);

# Compute EMD between XPts and each segment of YPts
nSeg = size(SegY, 1);
emd = Array{Float64}(nSeg);
for i in 1:nSeg
    (emd[i], ~, ~, ~) = wassersteindiscrete(
        xpoints = XPts,
        ypoints = YPts[colon(SegY[i, 1], SegY[i, 2]), :];
        solver = solver);
end

# Write output to MAT file
file = matopen(matOut, "w");
write(file, "emd", emd);
close(file);
