#=
# [04-denoise-1d](@id 04-denoise-1d)

## Basic Introduction to Machine Learning: 04-denoise-1d  

Illustrate 1D signal denoising using Julia's Flux library

- Jeff Fessler, University of Michigan  
- 2018-10-23 Julia 1.0.1 version
- 2023-01-29 Julia 1.8.5 version

This page was generated from a single Julia file:
[04-denoise-1d.jl](@__REPO_ROOT_URL__/04-denoise-1d.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](https://nbviewer.org/) here:
#md # [`04-denoise-1d.ipynb`](@__NBVIEWER_ROOT_URL__/04-denoise-1d.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`04-denoise-1d.ipynb`](@__BINDER_ROOT_URL__/04-denoise-1d.ipynb).


# ### Setup

# Packages needed here.

using LinearAlgebra: norm
using Random: seed!
using Distributions: Normal
import Flux # Julia package for deep learning
using Flux: throttle, mse
using LaTeXStrings # pretty plot labels
using Plots: plot, plot!, scatter, scatter!, histogram, default, font, gui
using MIRTjim: jim, prompt
using InteractiveUtils: versioninfo

default(markersize=5, markerstrokecolor=:auto, label="")
#fnt = font("DejaVu Sans", 15) # larger default font
#default(guidefont=fnt, xtickfont=fnt, ytickfont=fnt, legendfont=fnt)
default(tickfontsize=10, legendfontsize=11)


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


#=
## Training data
Generate training and testing data; 1D piece-wise constant signals
=#

# Function to generate a random piece-wise constant signal
function makestep(; dim=32, njump=3, valueDist=Normal(0,1), minsep=2)
    jump_locations = randperm(dim)[1:njump]
    while minimum(diff(jump_locations)) <= minsep
        jump_locations = randperm(dim)[1:njump] # random jump locations
    end
    index = zeros(dim)
    index[jump_locations] .= 1
    index = cumsum(index)
    values = rand(valueDist, njump) # random signal values
    x = zeros(dim)
    for jj=1:njump
        x[index .== jj] .= values[jj]
    end
    x[index .== 0] .= values[njump] # periodic end conditions
    x = circshift(x, rand(1:dim, 1)) # random shift - just to be sure
    return x
end

# Training data
seed!(0)
siz = 32

ntrain = 2^11
Xtrain = [makestep(dim=siz) for _ in 1:ntrain] # noiseless data
Xtrain = hcat(Xtrain...) # (siz, ntrain)

ntest = 2^10
Xtest = [makestep(dim=siz) for _ in 1:ntest] # noiseless data
Xtest = hcat(Xtest...) # (siz, ntest)

p0 = plot(Xtrain[:,1:14], label="")

#
prompt()


# Data covariance
Kx = Xtrain*Xtrain' / ntrain
p1 = jim(Kx, title="Kx");

# Add noise
σnoise = 0.3
Ytrain = Xtrain + σnoise * randn(size(Xtrain)) # noisy train data
Ytest = Xtest + σnoise * randn(size(Xtest)) # noisy test data
Ky = Ytrain*Ytrain' / ntrain;

@show maximum(Kx)
@show maximum(Ky)

p2 = jim(Ky, title="Ky")
plot(p0, p1, p2, layout=(3,1))

#
prompt()


#=
Wiener filter (MMSE estimator)
=#

## cond(Kx + σnoise*I)
Hw = Kx * inv(Kx + σnoise^2 * I)
jim(Hw')


# Denoise via Wiener filter (MMSE *linear* method)
Xw = Hw * Ytest
nrmse = (Xh) -> round(norm(Xh - Xtest) / norm(Xtest) * 100, digits=2)
@show nrmse(Ytest), nrmse(Xw)
plot(Xw[:,1:4])
plot!(Xtest[:,1:4])

#
prompt()


# Verify that marginal distribution is Gaussian
histogram(Xtrain[:], label = "Xtrain hist")

#=
## Simple NN

Examine a "NN" that is a single fully connected affine layer  
(It should perform same as Wiener filter.)

First try a basic affine NN model
=#
model1 = Chain(Dense(siz,siz))
loss(model, x, y) = mse(model(x), y)

iters = 2^12
dataset = Base.Iterators.repeated((Ytrain, Xtrain), iters) # trick X,Y swap for denoising!

throw() # todo to here

state1 = Flux.setup(loss, dataset, ADAM(Flux.params(model1)))
Flux.train!(loss, dataset, ADAM(Flux.params(model1)))

# +
# compare training affine NN to wiener filter
H1 = model1(Matrix(I, siz, siz)).data
heatmap(H1, aspect_ratio=1, yflip=true, title="Learned filter for affine NN")

X1 = H1 * Ytest
X1nn = model1(Ytest).data
@show nrmse(Ytest), nrmse(Xw), nrmse(X1), nrmse(X1nn)
bias = model1(zeros(siz)).data
@show minimum(bias), maximum(bias)
plot!()
# -

# ### Examine a single hidden layer NN

# +
# create a basic NN model
nhidden = siz*2 # neurons in hidden layer
model2 = Chain(Dense(siz,nhidden,relu), Dense(nhidden,siz))
loss(x, y) = mse(model2(x), y)

iters = 2^12
dataset = Base.Iterators.repeated((Ytrain, Xtrain), iters) # trick X,Y swap for denoising!

evalcb = () -> @show(loss(Ytest,Xtest).data)
cb = throttle(evalcb, 4)

Flux.train!(loss, dataset, ADAM(Flux.params(model2)), cb=cb)
# -

X2 = model2(Ytest).data
@show nrmse(Ytest), nrmse(Xw), nrmse(X1), nrmse(X1nn), nrmse(X2)

# Examine joint distribution
lag = 1
tmp = circshift(Xtrain, (lag,))
histogram2d(Xtrain[:], tmp[:])
plot!(aspect_ratio=1, xlim=[-4,4], ylim=[-4,4])
plot!(xlabel=L"x[n]", ylabel=latexstring("x[n-$lag \\ mod \\ N]"))
plot!(title="Joint histogram of neighbors")



# ### Experiments below here - work in progress  
# [https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl]

# +
model3 = Chain(
  Conv((2,2), 1=>16, relu),
  x -> maxpool(x, (2,2)),
  Conv((2,2), 16=>8, relu),
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)))
#  Dense(288, 10), softmax)

tmp = Xtrain[:,1:40];
#model3(tmp)
# -

tmp = Conv((3,1), 1=>1, relu)
model3 = Chain(tmp)
#model3 = Chain(Dense(siz,nhidden,relu))
tmp = reshape(Xtrain[:,1], siz, 1);
#model3(tmp)

tmp = Conv((3,3), 1=>1, relu)
model3 = Chain(tmp)
#model3 = Chain(Dense(siz,nhidden,relu))
tmp = reshape(Xtrain[:,1], siz, 1)
tmp = Xtrain[:,1:4];
#model3(tmp)

# +
# create a basic NN model
nhidden = siz*2 # neurons in hidden layer
model3 = Chain(Dense(siz,nhidden,relu), Dense(nhidden,siz))
loss(x, y) = mse(model2(x), y)

iters = 2^12
dataset = Base.Iterators.repeated((Ytrain, Xtrain), iters) # trick X,Y swap for denoising!

evalcb = () -> @show(loss(Ytest,Xtest).data)
cb = throttle(evalcb, 4);

#Flux.train!(loss, dataset, ADAM(Flux.params(model2)), cb=cb)


# ### Reproducibility

# This page was generated with the following version of Julia:
io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')

# And with the following package versions
import Pkg; Pkg.status()
