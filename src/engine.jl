import Base

mutable struct Value
    v::Float64
    grad::Float64
    xs::Array{Value}
    op::Symbol
end

Value(v::Number; grad=0., xs=[], op=:atom) = Value(Float64(v), grad, xs, op)

Base.:+(a::Value, b::Value) = Value(a.v + b.v; xs=[a, b], op=:+)
Base.:*(a::Value, b::Value) = Value(a.v * b.v; xs=[a, b], op=:*)
Base.:^(a::Value, b::Value) = Value(a.v ^ b.v; xs=[a, b], op=:^)
relu(a::Value) = Value(max(0, a.v); xs=[a], op=:relu)

# avoid writing stuffs like a^(0-2)
Base.inv(a::Value) = a ^ (0-1)

Base.:+(a::Value, b::Number) = a + Value(Float64(b))
Base.:*(a::Value, b::Number) = a * Value(Float64(b))
Base.:^(a::Value, b::Number) = a ^ Value(Float64(b))

Base.:+(a::Number, b::Value) = Value(Float64(a)) + b
Base.:*(a::Number, b::Value) = Value(Float64(a)) * b
Base.:^(a::Number, b::Value) = Value(Float64(a)) ^ b

Base.:+(a::Value) = a * 1
Base.:-(a::Value) = a * -1

Base.:-(a::Union{Value, Number}, b::Union{Value, Number}) = a + -b
Base.:/(a::Union{Value, Number}, b::Union{Value, Number}) = a * b^-1
Base.:\(a::Union{Value, Number}, b::Union{Value, Number}) = a^-1 * b

Base.show(io::IO, v::Value) = print(io, "Value($(v.v), grad=$(v.grad))")

function build_topo(v)
    topo, visited = [], Set()
    function build(x)
        if !(x in visited)
            push!(visited, x)
            build.(x.xs)
            push!(topo, x)
        end
    end
    build(v)
    reverse(topo)
end

function update!(v::Value)
    if v.op == :+
        @assert length(v.xs) == 2
        a, b = v.xs
        a.grad += v.grad
        b.grad += v.grad
    elseif v.op == :*
        @assert length(v.xs) == 2
        a, b = v.xs
        a.grad += b.v * v.grad
        b.grad += a.v * v.grad
    elseif v.op == :^
        @assert length(v.xs) == 2
        a, b = v.xs
        a.grad += a.v^(b.v-1) * b.v * v.grad
    elseif v.op == :relu
        @assert length(v.xs) == 1
        a, = v.xs
        a.grad += v.v > 0 ? v.grad : 0.
    end
    v
end

function backward!(v::Value)
    v.grad = 1.
    update!.(build_topo(v))
    v
end
