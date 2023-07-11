module MicroGrad

include("engine.jl")
export Value, relu, backward!

include("nn.jl")
export Neuron, Layer, MLP, parameters, zero_grad!

end # module micrograd
