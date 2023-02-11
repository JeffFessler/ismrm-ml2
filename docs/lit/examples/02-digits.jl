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
using StatsBase: mean
using MLDatasets: MNIST
using Random: seed!, randperm
using LaTeXStrings # pretty plot labels
using Plots: default, gui, plot, scatter, plot!, scatter!
using MIRTjim: jim, prompt
using InteractiveUtils: versioninfo
default(markersize=5, markerstrokecolor=:auto, label="")

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Load data

Read the MNIST data for some handwritten digits.
This code will automatically download the data from web if needed
and put it in a folder like: `~/.julia/datadeps/MNIST/`.
=#
if !@isdefined(data)
    digitn = (4, 9) # which digits to use
    isinteractive() || (ENV["DATADEPS_ALWAYS_ACCEPT"] = true) # avoid prompt
    dataset = MNIST(Float32, :train)
    nrep = 1000
    ## function to extract the 1st 1000 examples of digit n:
    data = n -> dataset.features[:,:,findall(==(n), dataset.targets)[1:nrep]]
    data = 255 * cat(dims=4, data.(digitn)...)
    nx, ny, nrep, ndigit = size(data)
    data = data[:,2:ny,:,:] # make images non-square to force debug
    ny = size(data,2)
    d0 = data[:,:,:,1]
    d1 = data[:,:,:,2]
    size(data)
end


# Look at sorted and unsorted images to show (un)supervised
seed!(0)
nrow = 4
ncol = 6
t0 = d0[:,:,1:nrow*ncol÷2]
t0[:,:,6] = d0[:,:,222] # include one ambiguous case
t1 = d1[:,:,1:nrow*ncol÷2]
tmp = cat(t0, t1, dims=3)
jim(tmp)
#src savefig("02-digit-rand.pdf")

tmp = tmp[:,:,randperm(size(tmp,3))] # for unsupervised
jim(tmp)
#src savefig("02-digit-sort.pdf")


# Use some data for training, and some for test
ntrain = 100
ntest = nrep - ntrain
train0 = d0[:,:,1:ntrain] # training data
train1 = d1[:,:,1:ntrain]
test0 = d0[:,:,(ntrain+1):end] # testing data
test1 = d1[:,:,(ntrain+1):end];

# SVD for singular vectors and low-rank subspace approximation
u0 = svd(reshape(train0, nx*ny, :)).U
u1 = svd(reshape(train1, nx*ny, :)).U
r0 = 3 # selected ranks
r1 = 3
q0 = reshape(u0[:,1:r0], nx, ny, :)
q1 = reshape(u1[:,1:r1], nx, ny, :)
p0 = jim(q0; nrow = 1)
p1 = jim(q1; nrow = 1)
p01 = plot(p0, p1, layout=(2,1))

#
prompt()

# ## How well do the first left singular vectors separate the two classes?

regress = (data, u) -> vec(mapslices(slice -> u'*slice[:], data, dims=(1,2)))
l1 = "$(digitn[1])"
l2 = "$(digitn[2])"
plot(xlabel = l1 * " U[:,1]", ylabel = l2 * " U[:,1]", legend=:topleft)
scatter!(regress(train0, u0[:,1]), regress(train0, u1[:,1]), label=l1)
scatter!(regress(train1, u0[:,1]), regress(train1, u1[:,1]), label=l2)

#
prompt()


# ## Classify test digits based on nearest subspace

Q0 = reshape(q0, nx*ny, r0)
Q1 = reshape(q1, nx*ny, r1);

y0 = reshape(test0, nx*ny, :)
y00 = y0 - Q0 * (Q0' * y0)
y01 = y0 - Q1 * (Q1' * y0)
correct0 = (mapslices(norm, y00, dims=1) .< mapslices(norm, y01, dims=1))
c0 = count(correct0) / ntest

#
y1 = reshape(test1, nx*ny, :)
y10 = y1 - Q0 * (Q0' * y1)
y11 = y1 - Q1 * (Q1' * y1)
correct1 = (mapslices(norm, y10, dims=1) .> mapslices(norm, y11, dims=1))
c1 = count(correct1) / ntest

# ### If I had more time I would show CNN-based classification here...


# ### Reproducibility

# This page was generated with the following version of Julia:
io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')

# And with the following package versions
import Pkg; Pkg.status()
