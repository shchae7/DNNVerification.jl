@with_kw struct NeurifyBaB <: Solver
    max_iter::Int64     = 100
    tree_search::Symbol = :DFS
    optimizer           = GLPK.optimizer
    ϵ::Float64          = 0.1
end


function solve(solver::NeurifyBaB, problem::Problem)
    isbounded(problem.input) || throw(UnboundedInputError("BaBNeurify can only handle bounded input sets."))

    nnet, output = problem.network, problem.output
    reach_list = []
    print(nnet)

    return CounterExampleResult(:unknown)
end