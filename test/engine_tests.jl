using Test
using Zygote
using .MicroGrad

function sanity_check_mg(x)
    x = Value(x)
    z = 2 * x + 2 + x
    q = relu(z) + z * x
    h = relu(z * z)
    y = h + q + q * x
    backward!(y)
    x.grad, y.v
end

function sanity_check_zg(x)
    relu(x) = max(0, x)
    y(x) = let
        z = 2 * x + 2 + x
        q = relu(z) + z * x
        h = relu(z * z)
        h + q + q * x
    end
    gradient(y, x)..., y(x)
end

function more_ops_mg(x, y)
    a = Value(x)
    b = Value(y)
    c = a + b
    d = a * b + b^3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + relu(b + a)
    d += 3 * d + relu(b - a)
    e = c - d
    f = e^2
    g = f / 2
    g += 10 / f
    backward!(g)
    [a.grad, b.grad, g.v]
end

function more_ops_zg(x, y)
    relu(x) = max(0, x)
    h(a, b) = let
        c = a + b
        d = a * b + b^3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + relu(b + a)
        d = d + 3 * d + relu(b - a)
        e = c - d
        f = e^2
        g = f / 2.0
        g + 10.0 / f
    end
    [gradient(h, x, y)..., h(x, y)]
end

@testset "Sanity check" begin
    @test sanity_check_mg(-4) == sanity_check_zg(-4)
end

@testset "More operations" begin
    @test isapprox(more_ops_mg(-4, 2), more_ops_zg(-4, 2))
end
