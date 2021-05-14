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


    return CounterExampleResult(:unknown)
end


# Always pick the first dom
function pick_out(doms)
    return (doms[1], doms[2:end])
end