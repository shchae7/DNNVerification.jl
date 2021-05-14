@with_kw struct NeurifyBaB <: Solver
    max_iter::Int64     = 100
    tree_search::Symbol = :DFS
    optimizer           = GLPK.Optimizer
    ϵ::Float64          = 0.1
end


function solve(solver::NeurifyBaB, problem::Problem)
    nnet, output = problem.network, problem.output
    println(nnet)

    reach_list = []

    domain = init_symbolic_grad(problem.input)
    println(domain)

    splits = Set()
    println(splits)


    reach = forward_network(solver, nnet, domain, collect=true)
    println(reach)

    i = 0
    j = 0
    aL, bL = reach[i].sym.Low[j, 1:end-1], reach[i].sym.Low[j, end]
    aU, bU = reach[i].sym.Up[j, 1:end-1], reach[i].sym.Up[j, end]
    println(aL)
    println(bL)
    println(aU)
    println(bU)

    return CounterExampleResult(:unknown)
end

# Symbolic forward_linear
function forward_linear(solver::Neurify, L::Layer, input::SymbolicIntervalGradient)
    output_Low, output_Up = interval_map(L.weights, input.sym.Low, input.sym.Up)
    output_Up[:, end] += L.bias
    output_Low[:, end] += L.bias
    sym = SymbolicInterval(output_Low, output_Up, domain(input))
    return SymbolicIntervalGradient(sym, input.LΛ, input.UΛ)
end

# Symbolic forward_act
function forward_act(solver::Neurify, L::Layer{ReLU}, input::SymbolicIntervalGradient)
    n_node = n_nodes(L)
    output_Low, output_Up = copy(input.sym.Low), copy(input.sym.Up)
    LΛᵢ, UΛᵢ = zeros(n_node), ones(n_node)
    # Symbolic linear relaxation
    # This is different from ReluVal
    for j in 1:n_node
        up_low, up_up = bounds(upper(input), j)
        low_low, low_up = bounds(lower(input), j)

        up_slope = relaxed_relu_gradient(up_low, up_up)
        low_slope = relaxed_relu_gradient(low_low, low_up)

        output_Up[j, :] .*= up_slope
        output_Up[j, end] += up_slope * max(-up_low, 0)

        output_Low[j, :] .*= low_slope

        LΛᵢ[j], UΛᵢ[j] = low_slope, up_slope
    end
    sym = SymbolicInterval(output_Low, output_Up, domain(input))
    LΛ = push!(input.LΛ, LΛᵢ)
    UΛ = push!(input.UΛ, UΛᵢ)
    return SymbolicIntervalGradient(sym, LΛ, UΛ)
end

function forward_act(solver::Neurify, L::Layer{Id}, input::SymbolicIntervalGradient)
    n_node = n_nodes(L)
    LΛ = push!(input.LΛ, ones(n_node))
    UΛ = push!(input.UΛ, ones(n_node))
    return SymbolicIntervalGradient(input.sym, LΛ, UΛ)
end



# Always pick the first dom
function pick_out(doms)
    return (doms[1], doms[2:end])
end