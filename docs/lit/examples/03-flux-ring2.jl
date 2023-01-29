#=
# [03-flux-ring2](@id 03-flux-ring2)

## Basic Introduction to Machine Learning: 03-flux-ring2

### Illustrate basic artificial NN training using Julia's Flux library

- Jeff Fessler, University of Michigan
- 2018-10-18 Julia 1.0.1 original
- 2023-01-29 Julia 1.8.5 update

This page was generated from a single Julia file:
[03-flux-ring2.jl](@__REPO_ROOT_URL__/03-flux-ring2.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](https://nbviewer.org/) here:
#md # [`03-flux-ring2.ipynb`](@__NBVIEWER_ROOT_URL__/03-flux-ring2.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`03-flux-ring2.ipynb`](@__BINDER_ROOT_URL__/03-flux-ring2.ipynb).


# ### Setup

# Packages needed here.

using LinearAlgebra: norm
using Random: seed!
using LaTeXStrings # pretty plot labels
import Flux # Julia package for deep learning
using Flux: Dense, Chain, relu, params, Adam, throttle, mse
using Plots: Plot, plot, plot!, scatter!, default, gui
using MIRTjim: jim, prompt
using InteractiveUtils: versioninfo

default(markersize=5, markerstrokecolor=:auto, label="")
default(legendfontsize=10, labelfontsize=12, tickfontsize=10)

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

# ## Generate (synthetic) data

# Function to simulate data that cannot be linearly separated
function simdata(; n1 = 40, n2 = 120, σ1 = 0.8, σ2 = 2, r2 = 3)
    data1 = σ1 * randn(2,n1)
    rad2 = r2 .+ σ2*rand(1,n2)
    ang2 = rand(1,n2) * 2π
    data2 = [rad2 .* cos.(ang2); rad2 .* sin.(ang2)]
    X = [data1 data2] # 2 × N = n1+n2
    Y = [-ones(1,n1) ones(1,n2)] # 1 × N
    @assert size(X,2) == size(Y,2)
    return (X,Y)
end;

# Scatter plot routine
function datasplit(X,Y)
    data1 = X[:,findall(==(-1), vec(Y))]
    data2 = X[:,findall(==(1), vec(Y))]
    return (data1, data2)
end;

function plot_data(X,Y; kwargs...)
    data1, data2 = datasplit(X,Y)
    plot(xlabel=L"x_1", ylabel=L"x_2"; kwargs...)
    scatter!(data1[1,:], data1[2,:], color=:blue, label="class1")
    scatter!(data2[1,:], data2[2,:], color=:red, label="class2")
    plot!(xlim=[-1,1]*6, ylim=[-1,1]*6)
    plot!(aspect_ratio=1, xtick=-6:6:6, ytick=-6:6:6)
end;


# Training data
seed!(0)
(Xtrain, Ytrain) = simdata()
plot_data(Xtrain,Ytrain)
#src savefig("ml-flux-data.pdf")

#
prompt()


# Validation and testing data
(Xvalid, Yvalid) = simdata()
(Xtest, Ytest) = simdata()

p1 = plot_data(Xvalid, Yvalid; title="Validation")
p2 = plot_data(Xtest, Ytest; title="Test")
plot(p1, p2)

#
prompt()


#=
## Train simple MLP model

A
[multilayer perceptron model]
(https://en.wikipedia.org/wiki/Multilayer_perceptron)
(MLP)
consists of multiple fully connected layers.

Train a basic NN model with 1 hidden layer
=#

if !@isdefined(state)
    nhidden = 10 # neurons in hidden layer
    model = Chain(Dense(2,nhidden,relu), Dense(nhidden,1))
    loss3(model, x, y) = mse(model(x), y) # admittedly silly choice
    iters = 10000
    dataset = Base.Iterators.repeated((Xtrain, Ytrain), iters)
    state = Flux.setup(Adam(), model)
    Flux.train!(loss3, model, dataset, state)
end;


# Plot results after training

function display_decision_boundaries(
    X, Y, model;
    x1range = range(-1,1,101)*6, x2range = x1range, τ = 0.0,
    kwargs...,
)
    data1,data2 = datasplit(X,Y)
    D = [model([x1;x2])[1] for x1 in x1range, x2 in x2range]
    jim(x1range, x2range, sign.(D.-τ); color=:grays, kwargs...)
    scatter!(data1[1,:], data1[2,:], color = :blue, label = "Class 1")
    scatter!(data2[1,:], data2[2,:], color = :red, label = "Class 2")
    plot!(xlabel=L"x_1", ylabel=L"x_2")
    plot!(xlim=[-1,1]*6, ylim=[-1,1]*6)
    plot!(aspect_ratio=1, xtick=-6:6:6, ytick=-6:6:6)
end;

# Examine classification accuracy
classacc(model, x, y::Number) = sign(model(x)[1]) == y
classacc(model, x, y::AbstractArray) = classacc(model, x, y[1])
function classacc(X, Y)
    tmp = zip(eachcol(X), eachcol(Y))
    tmp = count(xy -> classacc(model, xy...), tmp)
    tmp = tmp / size(Y,2) * 100
    return round(tmp, digits=3)
end

lossXY = loss3(model, Xtrain, Ytrain)
display_decision_boundaries(Xtrain, Ytrain, model)
plot!(title = "Train: MSE Loss = $(round(lossXY,digits=4)), " *
    "Class=$(classacc(Xtrain, Ytrain)) %")
#src savefig("ml-flux-final.pdf")

#
prompt()


#=
## Train while validating

Create a basic NN model with 1 hidden layer.
This version evaluates performance every epoch
for both the training data and validation data.
=#

nhidden = 10 # neurons in hidden layer
layer2 = Dense(2, nhidden, relu)
layer3 = Dense(nhidden, 1)
model = Chain(layer2, layer3)
loss3(model, x, y) = mse(model(x), y)

nouter = 80 # of outer iterations, for showing loss
losstrain = zeros(nouter+1)
lossvalid = zeros(nouter+1)

iters = 100
losstrain[1] = loss3(model, Xtrain, Ytrain)
lossvalid[1] = loss3(model, Xvalid, Yvalid)

for io in 1:nouter
    ## @show io
    idataset = Base.Iterators.repeated((Xtrain, Ytrain), iters)
    istate = Flux.setup(Adam(), model)
    Flux.train!(loss3, model, idataset, istate)
    losstrain[io+1] = loss3(model, Xtrain, Ytrain)
    lossvalid[io+1] = loss3(model, Xvalid, Yvalid)
    if (io ≤ 6) && false # set to true to make images
        display_decision_boundaries(Xtrain, Ytrain, model)
        plot!(title="$(io*iters) epochs")
        ## savefig("ml-flux-$(io*iters).pdf")
    end
end

loss_train = loss3(model, Xtrain, Ytrain)
loss_valid = loss3(model, Xvalid, Yvalid)
p1 = display_decision_boundaries(Xtrain, Ytrain, model;
 title="Train:\nMSE Loss = $(round(loss_train,digits=4))\n" *
    "Class=$(classacc(Xtrain, Ytrain)) %",
)
p2 = display_decision_boundaries(Xvalid, Yvalid, model;
 title="Valid:\nMSE Loss = $(round(loss_valid,digits=4))\n" *
    "Class=$(classacc(Xvalid, Yvalid)) %",
)
p12 = plot(p1, p2)

#
prompt()


# Show MSE loss vs epoch
ivalid = findfirst(>(0), diff(lossvalid))
plot(xlabel="epoch/$(iters)", ylabel="RMSE loss", ylim=[0,1.05*maximum(losstrain)])
plot!(0:nouter, sqrt.(losstrain), label="training loss", marker=:o, color=:green)
plot!(0:nouter, sqrt.(lossvalid), label="validation loss", marker=:+, color=:violet)
plot!(xticks = [0, ivalid, nouter])
#src savefig("ml-flux-loss.pdf")

#
prompt()


# Show response of (trained) first hidden layer
x1range = range(-1,1,31) * 6
x2range = range(-1,1,33) * 6
layer2data = [layer2([x1;x2])[n] for x1 = x1range, x2 = x2range, n in 1:nhidden]

pl = Array{Plot}(undef, nhidden)
for n in 1:nhidden
    ptmp = jim(x1range, x2range, layer2data[:,:,n], color=:cividis,
        xtick=-6:6:6, ytick=-6:6:6,
    )
    if n == 7
        plot!(ptmp, xlabel=L"x_1", ylabel=L"x_2")
    end
    pl[n] = ptmp
end
plot(pl[1:9]...)
#src savefig("ml-flux-layer2.pdf")

#
prompt()


# ### Reproducibility

# This page was generated with the following version of Julia:
io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')

# And with the following package versions
import Pkg; Pkg.status()
