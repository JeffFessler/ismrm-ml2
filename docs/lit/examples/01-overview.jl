#=
# [01-overview](@id 01-overview)

## Basic Introduction to Machine Learning: 01-overview  

This page was generated from a single Julia file:
[01-overview.jl](@__REPO_ROOT_URL__/01-overview.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](https://nbviewer.org/) here:
#md # [`01-overview.ipynb`](@__NBVIEWER_ROOT_URL__/01-overview.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`01-overview.ipynb`](@__BINDER_ROOT_URL__/01-overview.ipynb).


# ### Setup

# Packages needed here.

using LinearAlgebra: norm
using Random: seed!
using LaTeXStrings # pretty plot labels
using Plots: plot, plot!, scatter, scatter!, surface!, default, font, gui
using MIRTjim: jim, prompt
using InteractiveUtils: versioninfo

default(markersize=5, markerstrokecolor=:auto, label="")
fnt = font("DejaVu Sans", 15) # larger default font
default(guidefont=fnt, xtickfont=fnt, ytickfont=fnt, legendfont=fnt)
default(tickfontsize=10, legendfontsize=11)


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);


# ## Supervised learning: classification
seed!(0)
n1 = 50; n2 = n1
rot = phi -> [cos(phi) sin(-phi); sin(phi) cos(phi)]
data1 = rot(π/8) * ([3 0; 0 1] * randn(2,n1) .+ [8;2])
data2 = rot(π/4) * ([2 0; 0 1] * randn(2,n2) .+ [9;3])
scatter(data1[1,:], data1[2,:], color=:blue, label="class1")
scatter!(data2[1,:], data2[2,:], color=:red, label="class2")
plot!(xlabel=L"x_1", ylabel=L"x_2")
plot!(xlim=(0,14), ylim=(0,14))
plot!(aspect_ratio=1, xtick=[0, 14], ytick=[0, 14])
#src savefig("ml-super1.pdf")
x = LinRange(0,14,101)
y = 2 .+ x - 0.03 * x.^2 # add decision boundary
plot!(x, y, color=:magenta)
#src savefig("ml-super2.pdf")

#
prompt()


# ## Supervised learning: regression
seed!(0)
N = 40
f = (x) -> 10. / (x + 1)
xt = 10 * rand(N)
yt = f.(xt) + 0.4 * randn(N)
x = range(0, 10, 101)
y = f.(x)
scatter(xt, yt, color=:blue, label="training data")
plot!(xlabel=L"x", ylabel=L"y")
#src plot!(x, y, line=:red)
plot!(xlim=(0,10), ylim=(0,8))
plot!(xtick=0:5:10, ytick=0:4:8)
#src savefig("ml-super-regress1.pdf")

# Polynomial regression model
Afun = (tt) -> [t.^i for t in tt, i in 0:3] # matrix of monomials
A = Afun(xt)
coef = A \ yt
y = Afun(x) * coef
plot!(x, y, line=:magenta, label="cubic regression")
#src savefig("ml-super-regress2.pdf")

#
prompt()


# ## Unsupervised learning: data
seed!(0)
n1 = 50; n2 = n1; n3=n1
rot = phi -> [cos(phi) sin(-phi); sin(phi) cos(phi)]
data1 = rot(π/4) * ([2 0; 0 0.7] * randn(2,n1) .+ [9;3])
data2 = rot(π/8) * ([3 0; 0 0.6] * randn(2,n2) .+ [8;2])
data3 = rot( 0 ) * ([2 0; 0 0.5] * randn(2,n3) .+ [9;1]);

plot(xlabel = L"x_1", ylabel = L"x_2")
scatter!(data1[1,:], data1[2,:], color=:black, label="training data")
scatter!(data2[1,:], data2[2,:], color=:black)
scatter!(data3[1,:], data3[2,:], color=:black)
plot!(xlim=(0,14), ylim=(0,14))
plot!(aspect_ratio=1, xtick=[0, 14], ytick=[0, 14])
#src savefig("ml-unsup1.pdf")

#
prompt()

# ## Clustering (oracle)
plot(xlabel = L"x_1", ylabel = L"x_2")
scatter!(data1[1,:], data1[2,:], color=:blue, label="cluster1")
scatter!(data2[1,:], data2[2,:], color=:red, label="cluster2")
scatter!(data3[1,:], data3[2,:], color=:orange, label="cluster3")
plot!(xlim=(0,14), ylim=(0,14))
plot!(aspect_ratio=1, xtick=[0, 14], ytick=[0, 14])
#src savefig("ml-unsup2.pdf")

#
prompt()


# ## Novelty detection
plot(xlabel=L"x_1", ylabel=L"x_2")
scatter!(data1[1,:], data1[2,:], color=:black)
scatter!(data2[1,:], data2[2,:], color=:black)
scatter!(data3[1,:], data3[2,:], color=:black)
scatter!([10], [11], color=:red)
plot!(xlim=(0,14), ylim=(0,14))
plot!(aspect_ratio=1, xtick=[0, 14], ytick=[0, 14])
#src savefig("ml-unsup3.pdf")

#
prompt()


# ### The utility of nonlinearity

# 1D plot supervised learning: classification
seed!(0)
n1 = 20; n2 = n1; n3 = n1
data1 = 1 * randn(2,n1) .+ 5
data2 = 1 * randn(2,n2) .+ 0
data3 = 1 * randn(2,n3) .+ (-5)
plot(xlabel=L"x_1", ylabel="")
scatter!(data1[1,:], zeros(n1), color=:blue, label="class1")
scatter!(data2[1,:], zeros(n2), color=:red, label="class2")
scatter!(data3[1,:], zeros(n3), color=:blue)
plot!(xlim=(-8,7), ylim=(-1,1))
plot!(xtick=-6:3:6, ytick=[])
plot!([1, 1]*2, [-1, 1], color=:orange)
#src savefig("ml-nonlin1.pdf")

#
prompt()


# A simple nonlinearity, abs(feature), allows linear separation
f = x -> abs(x)
data1[2,:] = f.(data1[1,:])
data2[2,:] = f.(data2[1,:])
data3[2,:] = f.(data3[1,:])
plot(xlabel=L"x_1", ylabel=L"x_2")
scatter!(data1[1,:], data1[2,:], color=:blue, label="class1")
scatter!(data2[1,:], data2[2,:], color=:red, label="class2")
scatter!(data3[1,:], data3[2,:], color=:blue)
plot!(xlim=(-8,7), ylim=(-1,10))
plot!(xtick=-6:3:6, ytick=0:5:10)
plot!([-1, 1]*8, [1, 1]*2.4, color=:orange, width=2, legend=:top)
#src savefig("ml-nonlin2.pdf")

#
prompt()


# ## 2D example
seed!(0)
n1 = 40; n2 = 120
data1 = randn(2,n1)
rad2 = 3 .+ 3*rand(1,n2)
ang2 = rand(1,n2) * 2π
data2 = [rad2 .* cos.(ang2); rad2 .* sin.(ang2)]
plot(xlabel=L"x_1", ylabel=L"x_2")
scatter!(data1[1,:], data1[2,:], color=:blue, label="class1")
scatter!(data2[1,:], data2[2,:], color=:red, label="class2")
plot!(xlim=[-1,1]*6, ylim=[-1,1]*6)
plot!(aspect_ratio=1, xtick=-6:6:6, ytick=-6:6:6)
#src savefig("ml-nonlin2d-flat.pdf")

#
prompt()

plot!([0, 1, 0, -1, 0]*3, [-1, 0, 1, 0, -1]*3, color=:orange, width=2)
#src savefig("ml-nonlin2d-flat2.pdf")

#
prompt()

# ## Nonlinear lifting into 3D
lift_fun = (x) -> sum(abs.(x), dims=1)
lift1 = [data1; lift_fun(data1)]
lift2 = [data2; lift_fun(data2)]
plot(xlabel=L"x_1", ylabel=L"x_2", zlabel=L"$x_3 = |x_1| + |x_2|$")
scatter!(lift1[1,:], lift1[2,:], lift1[3,:], color=:blue, label="class1")
scatter!(lift2[1,:], lift2[2,:], lift2[3,:], color=:red, label="class2")
plot!(xlim=[-1,1]*6, ylim=[-1,1]*6)
plot!(xtick=-6:6:6, ytick=-6:6:6)
plot!(camera=(30,12))
#savefig("ml-nonlin2d-lift.pdf")

#
prompt()


xc = -6:6
yc = -6:6
z = 3 * ones(length(xc), length(yc)) # 3 chosen manually
surface!(xc, yc, z, colorbar=nothing, alpha=0.6) #, color=:orange)
#src savefig("ml-nonlin2d-lift2.pdf")

#
prompt()


# ## Nonlinearity in regression
seed!(0)
N = 40
f = (x) -> 10. / (x + 1)
xt = 10 * rand(N)
yt = f.(xt) + 0.4 * randn(N)
x = range(0, 10, 101)
y = f.(x)
scatter(xt, yt, color=:blue, label="training data for regression")
plot!(xlabel=L"x", ylabel=L"y")
#src plot!(x, y, line=:red)
plot!(xlim=(0,10), ylim=(0,8))
plot!(xtick=0:5:10, ytick=0:4:8)
#src savefig("ml-super-regress1.pdf")

Afun = (tt,deg) -> [t.^i for t in tt, i in 0:deg] # matrix of monomials
A3 = Afun(xt,3)
coef3 = A3 \ yt
y3 = Afun(x,3) * coef3;

A1 = Afun(xt,1)
coef1 = A1 \ yt
y1 = Afun(x,1) * coef1;

plot!(x, y3, line=:magenta,
    label = L"\mathrm{cubic:\ } y = \alpha_3 x^3 + \alpha_2 x^2 + \alpha_1 x + \alpha_0")
plot!(x, y1, line=(:dash,:red),
    label = L"\mathrm{linear\ (affine):\ } y = \alpha_1 x + \alpha_0")
#src savefig("ml-nonlin3-regress.pdf")

#
prompt()


# ## Linear discriminant analysis (LDA)  
# [LDA](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)

seed!(0)
n1 = 70; n2 = n1
rot = phi -> [cos(phi) sin(-phi); sin(phi) cos(phi)]
mu1 = [7, 10]
mu2 = [9, 4]
S1 = rot(π/9) * [3 0; 0 1]
S2 = S1 # for LDA
data1 = S1 * randn(2,n1) .+ mu1
data2 = S2 * randn(2,n2) .+ mu2
plot(xlabel=L"x_1", ylabel=L"x_2")
scatter!(data1[1,:], data1[2,:], color=:blue, label="class1")
scatter!(data2[1,:], data2[2,:], color=:red, label="class2")
plot!(xlim=(0,16), ylim=(0,16))
plot!(aspect_ratio=1)
plot!(xtick=0:4:16, ytick=0:4:16)

ϕ = range(0,2π,101)
for r in [1.5 2.5]
    local x = r * cos.(ϕ)
    local y = r * sin.(ϕ)
    c1 = S1 * [x'; y'] .+ mu1
    c2 = S2 * [x'; y'] .+ mu2
    plot!(c1[1,:], c1[2,:], color=:blue)
    plot!(c2[1,:], c2[2,:], color=:red)
end
x = range(-1,17,11)
w = (S1 * S1') \ (mu2 - mu1) # LDA
c = (norm(S1 \ mu2)^2 - norm(S1 \ mu1)^2)/2
y = (c .- w[1] * x) / (w[2])
plot!(x, y, color=:magenta, width=2, legend=:topleft)
#src savefig("ml-lda1.pdf")

#
prompt()


# ### Model-order selection

# Sinusoidal regression training data
seed!(0)
Ntrain = 40
Ntest = 30
f = (x) -> 10. / (x + 1)
xtrain = 10 * rand(Ntrain)
ytrain = f.(xtrain) + 0.4 * randn(Ntrain)
xtest = 10 * rand(Ntest)
ytest = f.(xtest) + 0.4 * randn(Ntest)

x = range(0,10,201)
y = f.(x)

plot(xlabel=L"x", ylabel=L"y")
scatter!(xtrain, ytrain, color=:blue, label="training data")
scatter!(xtest, ytest, color=:red, label="test data")
plot!(xlim=(0,10), ylim=(0,8))
plot!(xtick=0:5:10, ytick=0:4:8)
#src savefig("ml-order0.pdf")

#
prompt()


# Show overfit
scatter(xtrain, ytrain, color=:blue, label="training data")
plot!(xlim=(0,10), ylim=(0,8))
plot!(xtick=0:5:10, ytick=0:4:8)

Afun = (tt,deg) -> [t.^i for t in tt, i in 0:deg] # matrix of monomials
Afun = (tt,deg) -> [cos(2π*t*i/20) for t in tt, i in 0:deg] # matrix of sinusoids
dlist = [2 9 20]
clist = (:magenta, :red, :orange)
for ii in 1:length(dlist)
    local deg = dlist[ii]
    local A = Afun(xtrain,deg)
    local coef = A \ ytrain
    local y = Afun(x,deg) * coef
    plot!(x, y, line=clist[ii], width=2, label="$deg harmonics")
end
plot!(xlabel=L"x", ylabel=L"y")
#src savefig("ml-order29.pdf")

#
prompt()


# Fit improves with more harmonics, of course
dlist = 0:30
etrain = zeros(length(dlist))
etest = zeros(length(dlist))
errs = zeros(length(dlist))
for ii in 1:length(dlist)
    deg = dlist[ii]
    Atrain = Afun(xtrain, deg)
    Atest = Afun(xtest, deg)
    ## @show cond(A'*A) # sinusoids is more stable than polynomials
    local coef = Atrain \ ytrain
    yh = Atrain * coef
    etrain[ii] = norm(yh - ytrain) 
    etest[ii] = norm(Atest * coef - ytest)
    errs[ii] = norm(yh - f.(xt))
end
scatter(dlist, etrain, color=:blue, label="fit to training data")
plot!(xlabel = "model order: # of sinusoids")
plot!(ylabel = L"\mathrm{fit:\ } ‖ \hat{y} - y ‖_2")
plot!(ylim=[0,13], ytick=[0,13])
#src savefig("ml-order-fit1.pdf")
#src scatter!(dlist, errs, color=:magenta, label="error to truth")
scatter!(dlist, etest, color=:red, label="fit to test data")
#src savefig("ml-order-fit2.pdf")

#
prompt()


# ## Cross-validation

Nlearn = Int(Ntrain / 2)
Nvalid = Ntrain - Nlearn
xlearn = xtrain[1:Nlearn]
ylearn = ytrain[1:Nlearn]
xvalid = xtrain[(Nlearn+1):Ntrain]
yvalid = ytrain[(Nlearn+1):Ntrain]

plot(xlabel=L"x", ylabel=L"y")
scatter!(xlearn, ylearn, color=:blue, label="training data (fitting)")
scatter!(xvalid, yvalid, color=:cyan, label="validation data (model selection)")
plot!(xlim=(0,10), ylim=(0,8))
plot!(xtick=0:5:10, ytick=0:4:8)
#src savefig("ml-valid0.pdf")

#
prompt()

# fit improves with more harmonics, of course
dlist = 0:20
elearn = zeros(length(dlist))
evalid = zeros(length(dlist))
etest = zeros(length(dlist))
errs = zeros(length(dlist))
for ii in 1:length(dlist)
    deg = dlist[ii]
    Alearn = Afun(xlearn, deg)
    Avalid = Afun(xvalid, deg)
    Atest = Afun(xtest, deg)
    ## @show cond(A'*A) # sinusoids is more stable than polynomials
    local coef = Alearn \ ylearn
    elearn[ii] = norm(Alearn * coef - ylearn)
    evalid[ii] = norm(Avalid * coef - yvalid)
    etest[ii] = norm(Atest * coef - ytest)
    errs[ii] = norm([Alearn; Avalid]*coef - f.([xlearn; xvalid]))
end
scatter(dlist, elearn, color=:blue, label="fit to training data")
scatter!(dlist, evalid, color=:cyan, label="fit to validation data")
plot!(xlabel = "model order: # of sinusoids")
plot!(ylabel = L"fit: \ ‖ \hat{y} - y ‖_2")
plot!(ylim=[0,13], ytick=[0,13])
#src dbest = findmin(evalid)[2]
dbest = findall(diff(evalid) .>= 0)[1] # find first increase in validation error
plot!(xtick=[0, dlist[dbest], 20])
#src plot!(legend=:top)
#src savefig("ml-valid-fit1.pdf")
#src scatter!(dlist, errs, color=:magenta, label="error to truth")
scatter!(dlist, etest, color=:red, label="fit to test data")
#src savefig("ml-valid-fit2.pdf")

#
prompt()


# ### Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
