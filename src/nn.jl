import Base

struct Neuron
    w
    b
    nonlin
end

Neuron(nin; nonlin=true) =
    Neuron([Value(rand() * 2 - 1) for _ in 1:nin], Value(0), nonlin)

function (n::Neuron)(x)
    act = sum((wi*xi for (wi,xi) in zip(n.w, x))) + n.b
    n.nonlin ? relu(act) : act
end

struct Layer
    neurons
end

Layer(nin, nout; kwargs...) = Layer([Neuron(nin; kwargs...) for _ in 1:nout])

function (l::Layer)(x)
    out = [n(x) for n in l.neurons]
    length(out) == 1 ? out[1] : out
end

struct MLP
    layers
end

function MLP(nin, nouts)
    sz = vcat([nin], nouts)
    MLP([Layer(sz[i], sz[i+1]; nonlin=i!=length(nouts))
        for i in 1:length(nouts)])
end

function (m::MLP)(x)
    map(m.layers) do l x = l(x) end
    x
end

parameters(n::Neuron) = vcat(n.w, n.b)

parameters(l::Layer) = [p for n in l.neurons for p in parameters(n)]

parameters(m::MLP) = [p for l in m.layers for p in parameters(l)]

function zero_grad!(m)
    map(parameters(m)) do p p.grad = 0. end
    m
end

Base.show(io::IO, n::Neuron) =
    print(io, string(n.nonlin ? "ReLU" : "Linear", "Neuron($(length(n.w)))"))

Base.show(io::IO, l::Layer) =
    print(io, string("Layer of [", join(repr.(l.neurons), ", "), "]"))

Base.show(io::IO, m::MLP) =
    print(io, string("MLP of [", join(repr.(m.layers), ", "),"]"))
