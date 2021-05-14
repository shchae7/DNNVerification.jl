@with_kw struct NeurifyBaB <: Solver
    max_iter::Int64     = 100
    tree_search::Symbol = :DFS
    optimizer           = GLPK.Optimizer
    ϵ::Float64          = 0.1
end


function solve(solver::NeurifyBaB, problem::Problem)
    isbounded(problem.input) || throw(UnboundedInputError("BaBNeurify can only handle bounded input sets."))

    nnet, output = problem.network, problem.output
    reach_list = []
    domain = init_symbolic_grad(problem.input)
    splits = Set()

    println("Domain")
    println(domain)

    for i in 1:solver.max_iter
        if i > 1
            domain, splits = select!(reach_list, solver.tree_search)
            print(splits)
        end
    end

    return CounterExampleResult(:unknown)
end

function select!(reach_list, tree_search)
    if tree_search == :BFS
        reach = popfirst!(reach_list)
    elseif tree_search == :DFS
        reach = pop!(reach_list)
    else
        throw(ArgumentError(":$tree_search is not a valid tree search strategy"))
    end
    return reach
end