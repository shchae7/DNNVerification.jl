@with_kw struct BaBNeurify <: Solver
    max_iter::Int64     = 100
    tree_search::Symbol = :DFS
    optimizer           = GLPK.Optimizer
    ϵ::Float64          = 0.1
end


function solve(solver::BaBNeurify, problem::Problem)
    return CounterExampleResult(:unknown)
end