"""
    BaBSR 

# Reference
https://github.com/oval-group/scaling-the-convex-barrier
"""

@with_kw struct BaBSR <: Solver
    ϵ::Float64 = 0.1
end

function solve(solver::BaBSR, problem::Problem)
    
end