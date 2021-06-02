# Reference
# 1. https://github.com/sisl/NeuralVerification.jl/blob/457a556197a9cf34c04e8d516de21f3e3641c2f6/test/runtime/runtime_tests.jl
# 2. https://github.com/sisl/NeuralVerification.jl/blob/457a556197a9cf34c04e8d516de21f3e3641c2f6/test/runtime/runtime_aux.jl
# Acceptable forms of I/O for each tool specified in above files

using DNNVerification, LazySets, Test, LinearAlgebra, GLPK
import DNNVerification: ReLU, Id

acas_file = "./networks/ACASXU_experimental_v2a_1_1.nnet"
acas_nnet = read_nnet(acas_file, last_layer_activation = Id());

# ACAS PROPERTY 5
b_lower = [ -0.3242742570,  0.0318309886, -0.4999998960 ,  -0.5 ,  -0.5      ]
b_upper = [ -0.3217850849,  0.0636619772 , -0.4992041213,  -0.2272727273,  -0.1666666667      ]

in_hyper  = Hyperrectangle(low = b_lower, high = b_upper)
inputSet = convert(HPolytope, in_hyper)

# output5 <= output 3
outputSet = HPolytope([HalfSpace([0.0, 0.0, -1.0, 0.0, 1.0], 0.0)])

problem_polytope_polytope_acas = Problem(acas_nnet, in_hyper, outputSet);

solver=Neurify()
println("$(typeof(solver)) - acas property5_3")
timed_result =@timed solve(solver, problem_polytope_polytope_acas)
println(" - Time: " * string(timed_result[2]) * " s")
println(" - Output: ")
println(timed_result[1])
println("")