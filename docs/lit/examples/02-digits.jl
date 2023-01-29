#=
# [02-digits](@id 02-digits)

## Basic Introduction to Machine Learning: 02-digits  

## Hand-written digit classification
- 2018-10-23 Jeff Fessler, University of Michigan  

This page was generated from a single Julia file:
[02-digits.jl](@__REPO_ROOT_URL__/02-digits.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](https://nbviewer.org/) here:
#md # [`02-digits.ipynb`](@__NBVIEWER_ROOT_URL__/02-digits.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`02-digits.ipynb`](@__BINDER_ROOT_URL__/02-digits.ipynb).


# ### Setup

# Packages needed here.

using LinearAlgebra: norm, svd
using StatBase: mean
using Random: seed!
using LaTeXStrings # pretty plot labels
using Plots: plot #, plot!, scatter, scatter!, surface!, default, font, gui
using MIRTjim: jim, prompt
using InteractiveUtils: versioninfo

#default(markersize=5, markerstrokecolor=:auto, label="")
#fnt = font("DejaVu Sans", 15) # larger default font
#default(guidefont=fnt, xtickfont=fnt, ytickfont=fnt, legendfont=fnt)
#default(tickfontsize=10, legendfontsize=11)


# ## Load data

# Read the MNIST data file for 0 and 1 digits
# download from web if needed
file0 = "data4"
file1 = "data9"
if !isfile(file0)
    download("http://cis.jhu.edu/~sachin/digit/" * file0, file0)
end
if !isfile(file1)
    download("http://cis.jhu.edu/~sachin/digit/" * file1, file1)
end

nx = 28 # original image size
ny = 28
nrep = 1000

d0 = Array{UInt8}(undef, (nx,ny,nrep))
read!(file0, d0) # load images

d1 = Array{UInt8}(undef, (nx,ny,nrep))
read!(file1, d1) # load images

iy = 2:ny
d0 = d0[:,iy,:] # Make images non-square to help debug
d1 = d1[:,iy,:]
ny = length(iy)

# Convert images to Float32 to avoid overflow errors
d0 = Array{Float32}(d0)
d1 = Array{Float32}(d1)

@show size(d0);
# -

# function to display mosaic of multiple images
function imshow3(x)
    tmp = permutedims(x, [1, 3, 2])
    tmp = reshape(tmp, :, ny)
    return heatmap(1:size(tmp,1), 1:ny, tmp,
        xtick=[1,nx], ytick=[1,ny], yflip=true,
        color=:grays, transpose=true, aspect_ratio=1)
end

# +
# look at sorted and unsorted images to show (un)supervised
Random.seed!(0)
nrow = 4
ncol = 6
t0 = d0[:,:,1:Int(nrow*ncol/2)]
t0[:,:,6] = d0[:,:,222] # include one ambiguous case
t1 = d1[:,:,1:Int(nrow*ncol/2)]
tmp = cat(t0, t1, dims=3)

tmp = tmp[:,:,randperm(size(tmp,3))] # for unsupervised

pl = []
for ii=1:nrow
    p = imshow3(tmp[:,:,(1:ncol) .+ (ii-1)*ncol])
    plot!(p, colorbar=:none)
    for jj=1:(ncol-1)
        c = :yellow # unsup
#       c = ii <= nrow/2 ? :blue : :red
        plot!([1; 1]*jj*nx, [1; ny], label="", color=c, xtick=[], ytick=[], axis=:off)
    end
    push!(pl, p)
end
plot(pl..., layout=(nrow,1))
#savefig("02-digit-rand.pdf")
#savefig("02-digit-sort.pdf")
# -

# use some data for training, and some for test
ntrain = 100
ntest = nrep - ntrain
train0 = d0[:,:,1:ntrain] # training data
train1 = d1[:,:,1:ntrain]
test0 = d0[:,:,(ntrain+1):end] # testing data
test1 = d1[:,:,(ntrain+1):end];

# svd for singular vectors and low-rank subspace approximation
u0, _, _ = svd(reshape(train0, nx*ny, :))
u1, _, _ = svd(reshape(train1, nx*ny, :))
r0 = 3 # selected ranks
r1 = 3
q0 = reshape(u0[:,1:r0], nx, ny, :)
q1 = reshape(u1[:,1:r1], nx, ny, :)
p0 = imshow3(q0)
p1 = imshow3(q1)
plot(p0, p1, layout=(2,1))

# ### Examine how well the first left singular vectors separate the two classes  

regress = (data, u) -> mapslices(slice -> u'*slice[:], data, dims=(1,2))[:]
 scatter(regress(train0, u0[:,1]), regress(train0, u1[:,1]), label=file0)
scatter!(regress(train1, u0[:,1]), regress(train1, u1[:,1]), label=file1)
plot!(xlabel = file0 * " U[:,1]", ylabel = file1 * " U[:,1]", legend=:topleft)

# ### Classify test digits based on nearest subspace  

# +
Q0 = reshape(q0, nx*ny, r0)
Q1 = reshape(q1, nx*ny, r1)

y0 = reshape(test0, nx*ny, :)
y00 = y0 - Q0 * (Q0' * y0)
y01 = y0 - Q1 * (Q1' * y0)
correct0 = (mapslices(norm, y00, dims=1) .< mapslices(norm, y01, dims=1))[:]
@show sum(correct0) / ntest

y1 = reshape(test1, nx*ny, :)
y10 = y1 - Q0 * (Q0' * y1)
y11 = y1 - Q1 * (Q1' * y1)
correct1 = (mapslices(norm, y10, dims=1) .> mapslices(norm, y11, dims=1))[:]
@show sum(correct1) / ntest
# -

# ### If I had more time I would show CNN-based classification here...



# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
