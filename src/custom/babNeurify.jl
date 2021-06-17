@with_kw struct BaBNeurify <: Solver
    max_iter::Int64     = 100
    tree_search::Symbol = :DFS
    optimizer           = GLPK.Optimizer
    ϵ::Float64          = 0.1
end


function solve(solver::BaBNeurify, problem::Problem)
    isbounded(problem.input) || throw(UnboundedInputError("BaBNeurify can only handle bounded input sets."))

    nnet, output = problem.network, problem.output
    reach_list = []
    domain = init_symbolic_grad(problem.input)
    splits = Set()
    for i in 1:solver.max_iter
        if i > 1
            domain, splits = select!(reach_list, solver.tree_search)
        end

        reach = forward_network(solver, nnet, domain, collect=true)
        if i < 2
            println(reach)
            println(reach[0][0])
            println(reach[0][0].sym.Low)
            println(reach[0][0].sym.Up)
        end

        popfirst!(reach)
        result, max_violation_con = check_inclusion(solver, nnet, last(reach).sym, output)   # 여기까지 수정 필요 없는 듯

        if result.status === :violated
            return result
        elseif result.status === :unknown
            subdomains = constraint_refinement(nnet, reach, max_violation_con, splits) # branch and bound
            for domain in subdomains
                push!(reach_list, (init_symbolic_grad(domain), copy(splits)))
            end
        end
        isempty(reach_list) && return CounterExampleResult(:holds)
    end
    return CounterExampleResult(:unknown)
end

function check_inclusion(solver::BaBNeurify, nnet::Network, reach::SymbolicInterval, output)
    input_domain = domain(reach)

    model = Model(solver); set_silent(model)
    x = @variable(model, [1:dim(input_domain)])
    add_set_constraint!(model, input_domain, x)

    max_violation = 0.0
    max_violation_con = nothing
    for (i, cons) in enumerate(constraints_list(output))
        a, b = cons.a, cons.b
        c = max.(a, 0)'*reach.Up + min.(a, 0)'*reach.Low

        @objective(model, Max, c * [x; 1] - b)
        optimize!(model)

        if termination_status(model) == OPTIMAL
            if compute_output(nnet, value(x)) ∉ output
                return CounterExampleResult(:violated, value(x)), nothing
            end

            viol = objective_value(model)
            if viol > max_violation
                max_violation = viol
                max_violation_con = a
            end
        else
            # TODO can we be more descriptive?
            error("No solution, please check the problem definition.")
        end
    end

    if max_violation > 0.0
        return CounterExampleResult(:unknown), max_violation_con
    else
        return CounterExampleResult(:holds), nothing
    end
end

function constraint_refinement(nnet::Network, reach::Vector{<:SymbolicIntervalGradient}, max_violation_con::AbstractVector{Float64}, splits)
    i, j, influence = get_max_nodewise_influence(nnet, reach, max_violation_con, splits)

    aL, bL = reach[i].sym.Low[j, 1:end-1], reach[i].sym.Low[j, end]
    aU, bU = reach[i].sym.Up[j, 1:end-1], reach[i].sym.Up[j, end]

    ∩ = (set, lc) -> HPolytope([constraints_list(set); lc])

    subsets = [domain(reach[1])] # all the reaches have the same domain, so we can pick [1]

    if !iszero(aL)
        subsets = subsets .∩ [HalfSpace(aL, -bL), HalfSpace(aL, -bL), HalfSpace(-aL, bL)]
    end
    if !iszero(aU)
        subsets = subsets .∩ [HalfSpace(aU, -bU), HalfSpace(-aU, bU), HalfSpace(-aU, bU)]
    end
    return filter(!isempty, subsets)
end

function get_max_nodewise_influence(nnet::Network, reach::Vector{<:SymbolicIntervalGradient}, max_violation_con::AbstractVector{Float64}, splits)

    LΛ, UΛ = reach[end].LΛ, reach[end].UΛ
    is_ambiguous_activation(i, j) = (0 < LΛ[i][j] < 1) || (0 < UΛ[i][j] < 1)

    LG = UG = max_violation_con
    i_max, j_max, influence_max = 0, 0, -Inf

    for i in reverse(1:length(nnet.layers))
        layer = nnet.layers[i]
        sym = reach[i].sym
        if layer.activation isa ReLU
            for j in 1:n_nodes(layer)
                if is_ambiguous_activation(i, j)
                    r = radius(sym, j)
                    influence = max(abs(LG[j]), abs(UG[j])) * r
                    if influence >= influence_max && (i, j, influence) ∉ splits
                        i_max, j_max, influence_max = i, j, influence
                    end
                end
            end
        end

        LG_hat = max.(LG, 0.0) .* LΛ[i] .+ min.(LG, 0.0) .* UΛ[i]
        UG_hat = min.(UG, 0.0) .* LΛ[i] .+ max.(UG, 0.0) .* UΛ[i]

        LG, UG = interval_map(layer.weights', LG_hat, UG_hat)
    end

    (i_max == 0 || j_max == 0) && error("Can not find valid node to split")

    push!(splits, (i_max, j_max, influence_max))

    return (i_max, j_max, influence_max)
end

function forward_linear(solver::BaBNeurify, L::Layer, input::SymbolicIntervalGradient)
    output_Low, output_Up = interval_map(L.weights, input.sym.Low, input.sym.Up)
    output_Up[:, end] += L.bias
    output_Low[:, end] += L.bias
    sym = SymbolicInterval(output_Low, output_Up, domain(input))
    return SymbolicIntervalGradient(sym, input.LΛ, input.UΛ)
end

function forward_act(solver::BaBNeurify, L::Layer{ReLU}, input::SymbolicIntervalGradient)
    n_node = n_nodes(L)
    output_Low, output_Up = copy(input.sym.Low), copy(input.sym.Up)
    LΛᵢ, UΛᵢ = zeros(n_node), ones(n_node)

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

function forward_act(solver::BaBNeurify, L::Layer{Id}, input::SymbolicIntervalGradient)
    n_node = n_nodes(L)
    LΛ = push!(input.LΛ, ones(n_node))
    UΛ = push!(input.UΛ, ones(n_node))
    return SymbolicIntervalGradient(input.sym, LΛ, UΛ)
end